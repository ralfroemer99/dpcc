import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import proxsuite
from qpth.qp import QPFunction

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
        self.normalizer = normalizer

        if normalizer is not None:
            if normalizer.normalizers['actions'].__class__.__name__ != 'LimitsNormalizer' or \
                normalizer.normalizers['observations'].__class__.__name__ != 'LimitsNormalizer':
                raise ValueError('Only LimitsNormalizer is supported!')

        if cost_dims is not None:
            costs = torch.ones(transition_dim, device=self.device) * 1e-3
            for idx in cost_dims:
                costs[idx] = 1
            self.Q = torch.diag(torch.tile(costs, (self.horizon, )))                    # Quadratic cost
        else:
            self.Q = torch.eye(transition_dim * horizon, device=self.device)                    # Quadratic cost

        if self.normalizer is not None:
            # normalizer = self.normalizer    # for simple testing
            normalizer = self.normalizer.normalizers['observations']
            x_max = normalizer.maxs
            x_min = normalizer.mins
            self.Q *= torch.diag(torch.tile(torch.tensor(x_max - x_min, device='cuda') ** 2, (self.horizon, )))
        
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

        for constraint_spec in constraint_list:
            if constraint_spec[0] == 'deriv':
                self.dynamic_constraints.constraint_list.append(constraint_spec)
            else:
                self.safety_constraints.constraint_list.append(constraint_spec)

        self.safety_constraints.build_matrices()
        self.dynamic_constraints.build_matrices()
        self.append_constraint(self.safety_constraints)
        self.append_constraint(self.dynamic_constraints)
        self.add_numpy_constraints()     

    def project(self, trajectory, constraints=None):
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
        # trajectory = trajectory.reshape(trajectory.shape[0], -1) if trajectory.ndim == 3 else trajectory.view(-1)
        # if trajectory.ndim == 2:        # From H x T to HT
        #     batch_size = 1
        #     trajectory = trajectory.view(-1)
        # else:      # From B x H x T to B x (HT)
        batch_size = trajectory.shape[0]
        trajectory = trajectory.reshape(trajectory.shape[0], -1)

        # Cost
        r = - trajectory @ self.Q

        # Constraints
        if self.solver == 'qpth':
            A, b, C, d = self.A, self.b, self.C, self.d
        else:
            A, b, C, d = self.A_np, self.b_np, self.C_np, self.d_np

        if self.skip_initial_state:
            s_0 = trajectory[:self.transition_dim] if batch_size == 1 else trajectory[0, :self.transition_dim]    # Current state
            if self.solver == 'proxsuite':
                s_0 = s_0.cpu().numpy()
            counter = 0
            for constraint in self.dynamic_constraints.constraint_list:
                if constraint[0] == 'deriv':
                    x_idx = int(constraint[1][0])
                    b[counter * self.horizon] = s_0[x_idx]
                    counter += 1

        # Add additional constraints (if any) --> TODO: For proxsuite, add them to the numpy matrices
        if constraints is not None:
            for constraint in constraints:
                constraint.build_matrices()
                if self.solver == 'qpth':
                    A = torch.cat([A, constraint.A], dim=0)
                    b = torch.cat([b, constraint.b], dim=0)
                    C = torch.cat([C, constraint.C], dim=0)
                    d = torch.cat([d, constraint.d], dim=0)     
                else:
                    A = np.concatenate([A, constraint.A.cpu().numpy()], axis=0)
                    b = np.concatenate([b, constraint.b.cpu().numpy()], axis=0)
                    C = np.concatenate([C, constraint.C.cpu().numpy()], axis=0)
                    d = np.concatenate([d, constraint.d.cpu().numpy()], axis=0) 

        # start_time = time.time()
        if self.solver == 'qpth':
            # Solve optimization problem with qpth solver
            sol = QPFunction()(self.Q, r, C, d, A, b)
            sol = sol.view(dims)
        else:
            # Solve optimization problem with proxsuite solver
            r_np = r.cpu().numpy()
            sol_np = np.zeros((batch_size, self.horizon * self.transition_dim), dtype=np.float32)
            if self.parallelize == False:
                qp = proxsuite.proxqp.dense.QP(self.horizon * self.transition_dim, self.A_np.shape[0], self.C_np.shape[0])
                # qp = proxsuite.proxqp.sparse.QP(self.horizon * self.transition_dim, self.A_np.shape[0], self.C_np.shape[0]) --> Does not work?
                for i in range(batch_size):
                    qp.init(self.Q_np, r_np[i], A, b, C, None, d)
                    qp.solve()
                    sol_np[i] = qp.results.x
            else:
                with ThreadPoolExecutor() as executor:
                    results = list(executor.map(solve_qp_proxsuite, range(batch_size), [self.Q_np]*batch_size, [r_np]*batch_size, [A]*batch_size, [b]*batch_size, 
                                                [C]*batch_size, [d]*batch_size, [self.horizon]*batch_size, [self.transition_dim]*batch_size))

                for i, result in enumerate(results):
                    sol_np[i] = result
            sol = torch.tensor(sol_np, device=self.device).reshape(dims)

        # print(f'Projection time {self.solver}:', time.time() - start_time)

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
                        # normalizer = self.normalizer    # for simple testing
                        normalizer = self.normalizer.normalizers['observations']
                        x_min = normalizer.mins[dim]
                        x_max = normalizer.maxs[dim]
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
                    # normalizer = self.normalizer    # for simple testing
                    normalizer = self.normalizer.normalizers['observations']
                    x_min = normalizer.mins
                    x_max = normalizer.maxs

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
                    # normalizer = self.normalizer    # for simple testing
                    normalizer = self.normalizer.normalizers['observations']
                    x_min = normalizer.mins[x_idx]
                    x_max = normalizer.maxs[x_idx]
                    dx_min = normalizer.mins[dx_idx]
                    dx_max = normalizer.maxs[dx_idx]
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
