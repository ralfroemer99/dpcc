import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import proxsuite
import gurobipy as gp
from qpth.qp import QPFunction
from scipy.optimize import minimize, Bounds

def solve_qp_proxsuite(i, Q_np, r_np, A, b, C, d, horizon, transition_dim):
    qp = proxsuite.proxqp.dense.QP(horizon * transition_dim, A.shape[0], C.shape[0])
    qp.init(Q_np, r_np[i], A, b, C, None, d)
    qp.solve()
    return qp.results.x


class Projector:

    def __init__(self, horizon, transition_dim, constraint_list=[], normalizer=None, dt=0.1,
                 cost_dims=None, skip_initial_state=True, diffusion_timestep_threshold=0.5,
                 device='cuda', solver='proxsuite', parallelize=False):
        self.horizon = horizon
        self.transition_dim = transition_dim
        self.dt = torch.tensor(dt, device=device)
        self.skip_initial_state = skip_initial_state
        # self.only_last = only_last
        self.diffusion_timestep_threshold = diffusion_timestep_threshold
        self.device = device
        self.solver = solver
        self.parallelize = parallelize
        
        # Determine whether to include actions in the projection
        if normalizer is None:
            self.normalizer = None
        elif transition_dim != normalizer.normalizers['observations'].maxs.size:
            self.normalizer = ProjectionNormalizer(observation_normalizer=normalizer.normalizers['observations'], 
                                                   action_normalizer=normalizer.normalizers['actions'])
        else:
            self.normalizer = ProjectionNormalizer(observation_normalizer=normalizer.normalizers['observations'])
        
        # ------------------- ONLY FOR TESTING PROJECTION ------------------
        # else:
        #     self.normalizer = normalizer

        # Quadratic cost
        if cost_dims is not None:
            costs = torch.ones(transition_dim, device=self.device) * 1e-3
            for idx in cost_dims:
                costs[idx] = 1
            self.Q = torch.diag(torch.tile(costs, (self.horizon, )))
        else:
            self.Q = torch.eye(transition_dim * horizon, device=self.device)

        if self.normalizer is not None:
            self.Q *= torch.diag(torch.tile(torch.tensor(self.normalizer.maxs - self.normalizer.mins, device=self.device) ** 2, (self.horizon, )))
        
        self.A = torch.empty((0, self.transition_dim * self.horizon), device=self.device)   # Equality constraints
        self.b = torch.empty(0, device=self.device)
        self.C = torch.empty((0, self.transition_dim * self.horizon), device=self.device)   # Inequality constraints
        self.d = torch.empty(0, device=self.device)
        self.lb = torch.empty(0, device=self.device)
        self.ub = torch.empty(0, device=self.device)

        self.safety_constraints = SafetyConstraints(horizon=horizon, transition_dim=transition_dim, normalizer=self.normalizer, 
                                                 skip_initial_state=self.skip_initial_state, device=self.device)
        self.dynamic_constraints = DynamicConstraints(horizon=horizon, transition_dim=transition_dim, normalizer=self.normalizer,
                                                      skip_initial_state=self.skip_initial_state, dt=self.dt, device=self.device)
        self.obstacle_constraints = ObstacleConstraints(horizon=horizon, transition_dim=transition_dim, normalizer=self.normalizer,
                                                        skip_initial_state=self.skip_initial_state, dt=self.dt, device=self.device)

        for constraint_spec in constraint_list:
            if constraint_spec[0] == 'deriv':
                self.dynamic_constraints.constraint_list.append(constraint_spec)
            elif constraint_spec[0] == 'lb' or constraint_spec[0] == 'ub' or constraint_spec[0] == 'eq' or constraint_spec[0] == 'ineq':
                self.safety_constraints.constraint_list.append(constraint_spec)
            elif constraint_spec[0] == 'sphere_inside' or constraint_spec[0] == 'sphere_outside':
                self.obstacle_constraints.constraint_list.append(constraint_spec)

        self.safety_constraints.build_matrices()
        self.dynamic_constraints.build_matrices()
        self.obstacle_constraints.build_matrices()
        self.append_constraint(self.safety_constraints)
        self.append_constraint(self.dynamic_constraints)
        self.add_numpy_constraints()     

    def project(self, trajectory, constraints=None, return_costs=False):
        """
            trajectory: np.ndarray of shape (batch_size, horizon, transition_dim) or (horizon, transition_dim)
            Solve an optimization problem of the form 
                \hat z =   argmin_z 1/2 z^T Q z + r^T z
                        subject to  Az  = b
                                    Cz <= d
            where z = (o_0, o_1, ..., o_{H-1}) is the trajectory in vector form. The matrices A, b, C, and d are defined by the dynamic and safety constraints.
                                    
        """
        
        dims = trajectory.shape

        # Reshape the trajectory to a batch of vectors (from B x H x T to B x (HT) or a vector (from H x T to HT)
        # if trajectory.ndim == 2:        # From H x T to HT
        #     batch_size = 1
        #     trajectory = trajectory.view(-1)
        # else:      # From B x H x T to B x (HT)
        batch_size = trajectory.shape[0]
        trajectory_reshaped = trajectory.reshape(trajectory.shape[0], -1)

        # Cost
        r = - trajectory_reshaped @ self.Q
        r_np = r.cpu().numpy()
        # if self.solver == 'proxsuite' or self.solver == 'gurobi':
        #     r = r.cpu().numpy()
        Q = self.Q if self.solver == 'qpth' else self.Q_np
        trajectory_np = trajectory_reshaped.cpu().numpy()

        # Constraints
        if self.solver == 'qpth':
            A, b, C, d = self.A, self.b, self.C, self.d
        else:
            A, b, C, d = self.A_np, self.b_np, self.C_np, self.d_np

        if self.skip_initial_state:
            # s_0 = trajectory_reshaped[:self.transition_dim] if batch_size == 1 else trajectory_reshaped[0, :self.transition_dim]    # Current state
            s_0 = trajectory_reshaped[0, :self.transition_dim]
            if self.solver == 'proxsuite' or self.solver == 'gurobi':
                s_0 = s_0.cpu().numpy()
            counter = 0
            for constraint in self.dynamic_constraints.constraint_list:
                if constraint[0] == 'deriv':
                    x_idx = int(constraint[1][0])
                    b[counter * self.horizon] = s_0[x_idx]
                    counter += 1

        # Add additional constraints (if any) --> TODO: For proxsuite, add them to the numpy matrices
        # if constraints is not None:
        #     for constraint in constraints:
        #         constraint.build_matrices()
        #         if self.solver == 'qpth':
        #             A = torch.cat([A, constraint.A], dim=0)
        #             b = torch.cat([b, constraint.b], dim=0)
        #             C = torch.cat([C, constraint.C], dim=0)
        #             d = torch.cat([d, constraint.d], dim=0)     
        #         else:
        #             A = np.concatenate([A, constraint.A.cpu().numpy()], axis=0)
        #             b = np.concatenate([b, constraint.b.cpu().numpy()], axis=0)
        #             C = np.concatenate([C, constraint.C.cpu().numpy()], axis=0)
        #             d = np.concatenate([d, constraint.d.cpu().numpy()], axis=0) 

        projection_costs = np.ones(batch_size, dtype=np.float32)
        # start_time = time.time()
        if self.solver == 'qpth':           # Solve optimization problem with qpth solver
            sol = QPFunction()(Q, r, C, d, A, b)
            sol = sol.view(dims)
        elif self.solver == 'proxsuite':    # Solve optimization problem with proxsuite solver
            sol_np = np.zeros((batch_size, self.horizon * self.transition_dim), dtype=np.float32)
            if self.parallelize == False:
                qp = proxsuite.proxqp.dense.QP(self.horizon * self.transition_dim, self.A_np.shape[0], self.C_np.shape[0])
                # qp = proxsuite.proxqp.sparse.QP(self.horizon * self.transition_dim, self.A_np.shape[0], self.C_np.shape[0]) --> Does not work?
                for i in range(batch_size):
                    qp.init(Q, r_np[i], A, b, C, None, d)
                    qp.solve()
                    sol_np[i] = qp.results.x
                    projection_costs[i] = 0.5 * sol_np[i] @ Q @ sol_np[i] + r_np[i] @ sol_np[i] + 0.5 * trajectory_np[i] @ Q @ trajectory_np[i]
            else:
                with ThreadPoolExecutor() as executor:
                    results = list(executor.map(solve_qp_proxsuite, range(batch_size), [Q]*batch_size, [r_np]*batch_size, [A]*batch_size, [b]*batch_size, 
                                                [C]*batch_size, [d]*batch_size, [self.horizon]*batch_size, [self.transition_dim]*batch_size))

                for i, result in enumerate(results):
                    sol_np[i] = result
                    projection_costs[i] = 0.5 * sol_np[i] @ Q @ sol_np[i] + r_np[i] @ sol_np[i] + 0.5 * trajectory_np[i] @ Q @ trajectory_np[i]
            sol = torch.tensor(sol_np, device=self.device).reshape(dims)
        elif self.solver == 'gurobi':   # Solve optimization problem with gurobi solver
            sol_np = np.zeros((batch_size, self.horizon * self.transition_dim), dtype=np.float32)     

            # Create model --> Put these things in the init method
            model = gp.Model('nonconvex_qp')
            model.setParam('MipFocus', 1)
            model.setParam('SolutionLimit', 1e5)
            model.Params.LogToConsole = 1

            # Add variables
            tau = model.addMVar(shape=(self.horizon * self.transition_dim), lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name='tau')
            
            # Add constraints
            # model.addConstr(A @ tau == b, name='eq_constraints')      # Dynamics constraints
            for constraint_idx in range(len(self.obstacle_constraints.P_list)):
                P = self.obstacle_constraints.P_list[constraint_idx]
                q = self.obstacle_constraints.q_list[constraint_idx]
                v = self.obstacle_constraints.v_list[constraint_idx]
                for t in range(1, self.horizon):                        # Obstacle constraints
                    start_idx = t * self.transition_dim
                    end_idx = (t + 1) * self.transition_dim
                    model.addConstr(tau[start_idx: end_idx] @ P @ tau[start_idx: end_idx] + q @ tau[start_idx: end_idx] <= v,
                                    name=f'obstacle_{t}')     
            
            for i in range(batch_size):
                # Cost function
                cost = 0.5 * tau @ Q @ tau + r_np[i] @ tau
                cost += (A @ tau - b) @ (A @ tau - b)
                model.setObjective(cost, gp.GRB.MINIMIZE)
                model.update()
                # Warm start
                for idx, v in enumerate(model.getVars()):
                    v.Start = trajectory_np[i][idx]
                model.update()

                # start_time = time.time()
                model.optimize()
                # print(f'Projection time for sample {i}:', time.time() - start_time)
                if model.Status == 2:
                    sol_np[i] = np.array(model.getAttr("X"))
                else:
                    sol_np[i] = trajectory_np[i]
            
            sol = torch.tensor(sol_np, device=self.device).reshape(dims)
        elif self.solver == 'scipy':    # Solve optimization problem with scipy solver
            sol_np = np.zeros((batch_size, self.horizon * self.transition_dim), dtype=np.float32)

            Q_double = Q.astype('double')
            A_double = A.astype('double')
            b_double = b.astype('double')
            C_double = C.astype('double')
            d_double = d.astype('double')
            r_np_double = r_np.astype('double')
            trajectory_np_double = trajectory_np.astype('double')
            # Constraints
            constraints = ()
            for constraint_idx in range(len(self.obstacle_constraints.P_list)):
                P = self.obstacle_constraints.P_list[constraint_idx]
                q = self.obstacle_constraints.q_list[constraint_idx]
                v = self.obstacle_constraints.v_list[constraint_idx]
                for t in range(1, self.horizon):                        # Obstacle constraints
                    start_idx = t * self.transition_dim
                    end_idx = (t + 1) * self.transition_dim
                    constraints += ({'type': 'ineq', 'fun': lambda x, start_idx=start_idx, end_idx=end_idx: -x[start_idx: end_idx] @ P @ x[start_idx: end_idx] - q @ x[start_idx: end_idx] + v,
                                     'jac': lambda x, start_idx=start_idx, end_idx=end_idx: np.concatenate([np.zeros(start_idx), -2 * P @ x[start_idx: end_idx] - q, np.zeros(len(x) - end_idx)])},)
                    # constraints += ({'type': 'ineq', 'fun': lambda x, start_idx=start_idx, end_idx=end_idx: -x[start_idx: end_idx] @ x[start_idx: end_idx] + 0.9,
                    #                  'jac': lambda x, start_idx=start_idx, end_idx=end_idx: np.concatenate([np.zeros(start_idx), -2 * x[start_idx: end_idx], np.zeros(len(x) - end_idx)])},)

            # constraints += ({'type': 'eq', 'fun': lambda x: -C_double @ x + d_double, 'jac': lambda x: -C_double},)
            constraints += ({'type': 'eq', 'fun': lambda x: A_double @ x - b_double, 'jac': lambda x: A_double},)
            for i in range(batch_size):
                # Cost
                cost_fun = lambda x: 0.5 * x @ Q_double @ x + r_np_double[i] @ x # + \
                    # (A_double @ x - b_double) @ (A_double @ x - b_double)
                jac_cost_fun = lambda x: Q_double @ x + r_np_double[i]
                # Constraints
                res = minimize(fun=cost_fun, 
                               x0=np.ones_like(trajectory_np_double[i]), 
                               constraints=constraints, 
                               method='SLSQP', 
                               jac=jac_cost_fun, 
                               bounds=Bounds(-5 * np.ones_like(trajectory_np_double[i]), 5 * np.ones_like(trajectory_np_double[i])),
                            #    tol=1e-6,
                               options={'maxiter': 1000, 'disp': False})

                sol_np[i] = res.x

            sol = torch.tensor(sol_np, device=self.device).reshape(dims)

        # print(f'Projection time {self.solver}:', time.time() - start_time)
        if return_costs:
            return sol, projection_costs    # only implemented for proxsuite and parallelize=False
        else:
            return sol

    def append_constraint(self, constraint):
        self.C = torch.cat([self.C, constraint.C], dim=0)
        self.d = torch.cat([self.d, constraint.d], dim=0)
        self.A = torch.cat([self.A, constraint.A], dim=0)
        self.b = torch.cat([self.b, constraint.b], dim=0)
        if constraint.__class__.__name__ == 'SafetyConstraints':
            self.A_safe, self.b_safe, self.C_safe, self.d_safe = constraint.A, constraint.b, constraint.C, constraint.d
        elif constraint.__class__.__name__ == 'DynamicConstraints':
            self.A_dyn, self.b_dyn, self.C_dyn, self.d_dyn = constraint.A, constraint.b, constraint.C, constraint.d

    def add_numpy_constraints(self):
        self.A_np = self.A.cpu().numpy()
        self.b_np = self.b.cpu().numpy()     
        self.C_np = self.C.cpu().numpy()
        self.d_np = self.d.cpu().numpy()
        self.Q_np = self.Q.cpu().numpy()


class Constraints:

    def __init__(self, horizon, transition_dim, normalizer=None, device='cuda'):
        self.horizon = horizon
        self.transition_dim = transition_dim
        self.normalizer = normalizer
        self.device = device

        self.A = torch.empty((0, self.transition_dim * self.horizon), device=device)
        self.b = torch.empty(0, device=device)
        self.C = torch.empty((0, self.transition_dim * self.horizon), device=device)
        self.d = torch.empty(0, device=device)

    def build_matrices(self):
        pass


class SafetyConstraints(Constraints):

    def __init__(self, skip_initial_state=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_initial_state = skip_initial_state
        self.constraint_list = []
        
    def build_matrices(self, constraint_list=None):
        """
            Input:
                constraint_list: listof constraints
                    e.g. [('lb', [-1.0, -inf, 0]), ('ub', [1.0, 2.0, inf]), ('eq', ([0, 1, 1], 1.5)), ('ineq': ([1, 0, 0], 0.5))] -->
                    x_0 in [-1, 1], x_1 in [-inf, 2], x_2 in [0, inf], 0 * x_0 + 1 * x_1 + 1 * x_2 = 1.5, 1 * x_0 + 0 * x_1 + 0 * x_2 <= 0.5,
                    where x_i is the i-th dimension of the state or state-action vectors
            The matrices have the following shapes:
                C: (horizon * n_bounds, transition_dim * horizon)
                d: (horizon * n_bounds)
                lb: horizon * transition_dim
            C consists of n_bounds blocks of shape (horizon, transition_dim * horizon), where each block corresponds to a 
            constraint (ub or lb) on a specific dimension. The block has a 1 or -1 at the corresponding dimension and time step.
        """

        if constraint_list is None:
            constraint_list = self.constraint_list
        else:
            self.constraint_list.extend(constraint_list)

        for constraint in constraint_list:
            type = constraint[0]
            bound = constraint[1]
            if type == 'lb' or type == 'ub':
                for dim in range(len(bound)):
                    bound = torch.tensor(bound, device=self.device)
                    if bound[dim] == -torch.inf or bound[dim] == torch.inf:
                        continue
                    
                    mat_append = torch.zeros(self.horizon, self.transition_dim * self.horizon, device=self.device)
                    vec_append = torch.zeros(self.horizon, device=self.device)

                    sign = 1 if type == 'ub' else -1
                    for t in range(self.horizon):
                        mat_append[t, t * self.transition_dim + dim] = sign
                        vec_append[t] = sign * bound[dim]
                        
                    if self.normalizer is not None:
                        x_min = self.normalizer.mins[dim]
                        x_max = self.normalizer.maxs[dim]
                        mat_append = mat_append * (x_max - x_min) / 2
                        vec_append = vec_append - sign * (x_min + x_max) / 2

                    if self.skip_initial_state:
                        mat_append = mat_append[1:]
                        vec_append = vec_append[1:]

                    self.C = torch.cat((self.C, mat_append), dim=0)
                    self.d = torch.cat((self.d, vec_append), dim=0)
                continue         

            # type == 'eq' or 'ineq'
            mat_append = torch.zeros(self.horizon, self.transition_dim * self.horizon, device=self.device)
            vec_append = torch.zeros(self.horizon, device=self.device)

            for i in range(self.horizon):
                if self.normalizer is not None:
                    x_min = self.normalizer.mins
                    x_max = self.normalizer.maxs

                    # Unnormalize the constraints. TODO: Extend to actions by checking if dim <= action_dim
                    # We have Cs <= d, where s is the unnormalized state vector. This is converted to C's_n <= d', where s_n is the normalized state vector,
                    # by using the fact that s = (s_n + 1) * (s_max - s_min) / 2 + s_min.
                    a = bound[0] * (x_max - x_min) / 2
                    b = bound[1] - bound[0] @ (x_max + x_min) / 2
                else:
                    a = bound[0]
                    b = bound[1]
                
                mat_append[i, i * self.transition_dim: (i + 1) * self.transition_dim] = torch.tensor(a, device=self.device)
                vec_append[i] = torch.tensor(b, device=self.device)

            if self.skip_initial_state:
                mat_append = mat_append[1:]
                vec_append = vec_append[1:]

            if type == 'eq':
                self.A = torch.cat((self.A, mat_append), dim=0)
                self.b = torch.cat((self.b, vec_append), dim=0)
            else:
                self.C = torch.cat((self.C, mat_append), dim=0)
                self.d = torch.cat((self.d, vec_append), dim=0)         

class DynamicConstraints(Constraints):
    def __init__(self, skip_initial_state=True, dt=0.02, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_initial_state = skip_initial_state
        self.dt = dt
        self.constraint_list = []
        
    def build_matrices(self, constraint_list=None):
        """
            Input:
                constraint_list: list of constraints
                    e.g. [('deriv', [0, 2]), ('deriv', [1, 3])] -->
                    x_0[t+1] = x_0[t] + self.dt * x_2[t], x_1[t+1] = x_1[t] + self.dt * x_3[t]                                      (explicit Euler) or
                    x_0[t+1] = x_0[t] + self.dt * (x_2[t] + x_2[t+1]) / 2, x_1[t+1] = x_1[t] + self.dt * (x_3[t] + x_3[t+1]) / 2    (variant of trapezoidal rule)
                    where x_i[t] is the i-th dimension of the state or state-action vectors at time t
            The matrices have the following shapes:
                C: (horizon * n_bounds, transition_dim * horizon)
                d: (horizon * n_bounds)
        """

        if constraint_list is None:
            constraint_list = self.constraint_list
        else:
            self.constraint_list.extend(constraint_list)
        
        for constraint in constraint_list:
            type = constraint[0]
            vals = constraint[1]
            if 'deriv' in type:
                x_idx = int(vals[0])
                dx_idx = int(vals[1])
            
                mat_append = torch.zeros(self.horizon - 1, self.transition_dim * self.horizon, device=self.device)
                vec_append = torch.zeros(self.horizon - 1, device=self.device)

                # Calculate multiplicative factors needed for normalization
                if self.normalizer is not None: 
                    x_min = self.normalizer.mins[x_idx]
                    x_max = self.normalizer.maxs[x_idx]
                    dx_min = self.normalizer.mins[dx_idx]
                    dx_max = self.normalizer.maxs[dx_idx]
                    x_diff = x_max - x_min
                    dx_diff = dx_max - dx_min
                    dx_sum = dx_max + dx_min

                for i in range(self.horizon - 1):
                    if self.normalizer is not None:
                        mat_append[i, i * self.transition_dim + x_idx] = 1 * x_diff
                        mat_append[i, i * self.transition_dim + dx_idx] = self.dt * dx_diff
                        mat_append[i, (i + 1) * self.transition_dim + x_idx] = -1 * x_diff
                        vec_append[i] = - dx_sum * self.dt
                    else:
                        mat_append[i, i * self.transition_dim + x_idx] = 1
                        mat_append[i, i * self.transition_dim + dx_idx] = self.dt
                        mat_append[i, (i + 1) * self.transition_dim + x_idx] = -1
                        vec_append[i] = 0

                if self.skip_initial_state:     # --> Do that in the projection method because it needs the current state. For that, record the relevant rows
                    mat_fix_initial = torch.zeros(1, self.transition_dim * self.horizon, device=self.device)    # Fix the initial state
                    mat_fix_initial[0, x_idx] = 1
                    mat_append = torch.cat((mat_fix_initial, mat_append), dim=0)
                    vec_append = torch.cat((torch.tensor([0], device=self.device), vec_append), dim=0)          # Must be changed to current state in each iteration!

                self.A = torch.cat((self.A, mat_append), dim=0)
                self.b = torch.cat((self.b, vec_append), dim=0)

class ObstacleConstraints(Constraints):
    def __init__(self, skip_initial_state=True, dt=0.02, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_initial_state = skip_initial_state
        self.dt = dt
        self.constraint_list = []

    def build_matrices(self, constraint_list=None):
        """
            Input:
                constraint_list: list of constraints
                    e.g. [('sphere_inside', [0, 2] [-1, 5], 1), ('sphere_outside', [1, 3], [0, 1], 4)] -->
                    (x_0 + 1)^2 + (x_2 - 5)^2 <= 1, x_1^2 + (x_3 - 1)^2 >= 4
                    where x_i is the i-th dimension of the state or state-action vectors at time t
            Generate the matrix P and the vector q for:
                s_i^T P s_i + q^T s_i <= v
            The matrices have the following shapes:
                C: (horizon * n_bounds, transition_dim * horizon)
                d: (horizon * n_bounds)
        """

        if constraint_list is None:
            constraint_list = self.constraint_list
        else:
            self.constraint_list.extend(constraint_list)

        self.P_list = []
        self.q_list = []
        self.v_list = []
        for constraint in constraint_list:
            type = constraint[0]
            dims = constraint[1]
            center = constraint[2]
            radius = constraint[3]

            P = np.zeros((self.transition_dim, self.transition_dim))
            q = np.zeros(self.transition_dim)
            v = radius ** 2

            dim_counter = 0
            for dim in dims:
                if self.normalizer is not None:
                    delta_s = self.normalizer.maxs[dim] - self.normalizer.mins[dim]
                    s_min = self.normalizer.mins[dim]
                    P[dim, dim] = delta_s ** 2 / 4
                    q[dim] = delta_s**2 / 2 + delta_s * (s_min - center[dim_counter])
                    v -= delta_s**2 / 4 + delta_s * (s_min - center[dim_counter]) + (s_min - center[dim_counter]) ** 2
                else:
                    P[dim, dim] = 1
                    q[dim] = -2 * center[dim_counter]
                    v -= center[dim_counter] ** 2
                dim_counter += 1

            if type == 'sphere_outside':
                P = -P
                q = -q
                v = -v

            self.P_list.append(P)
            self.q_list.append(q)
            self.v_list.append(v)    


class ProjectionNormalizer():
    def __init__(self, observation_normalizer=None, action_normalizer=None):
        self.observation_normalizer = observation_normalizer
        self.action_normalizer = action_normalizer
        self.get_limits()

    def get_limits(self):
        if self.observation_normalizer is not None and self.action_normalizer is not None:
            x_max = np.concatenate([self.action_normalizer.maxs, self.observation_normalizer.maxs])
            x_min = np.concatenate([self.action_normalizer.mins, self.observation_normalizer.mins])
        elif self.observation_normalizer is not None:
            x_max = self.observation_normalizer.maxs
            x_min = self.observation_normalizer.mins
        elif self.action_normalizer is not None:
            x_max = self.action_normalizer.maxs
            x_min = self.action_normalizer.mins
        
        self.maxs = x_max
        self.mins = x_min
                