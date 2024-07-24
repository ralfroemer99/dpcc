import torch
from qpth.qp import QPFunction
from casadi import nlpsol

class Projector:

    def __init__(self, horizon, transition_dim, constraint_list=[], normalizer=None, dt=0.1, 
                 skip_initial_state=True, only_last=False, device='cuda'):
        self.horizon = horizon
        self.transition_dim = transition_dim
        self.dt = torch.tensor(dt, device=device)
        self.skip_initial_state = skip_initial_state
        self.only_last = only_last
        self.device = device

        self.normalizer = normalizer

        # if normalizer is not None:
        #     if normalizer.normalizers['actions'].__class__.__name__ != 'LimitsNormalizer' or \
        #         normalizer.normalizers['observations'].__class__.__name__ != 'LimitsNormalizer':
        #         raise ValueError('Normalizer not supported')

        if self.normalizer is not None:
            # normalizer = self.normalizer    # for simple testing
            normalizer = self.normalizer.normalizers['observations']
            x_max = normalizer.maxs
            x_min = normalizer.mins
            self.Q = torch.diag(torch.tile(torch.tensor(x_max - x_min, device='cuda') ** 2, (self.horizon, )))
        else:
            self.Q = torch.eye(transition_dim * horizon, device=self.device)                    # Quadratic cost
        self.A = torch.empty((0, self.transition_dim * self.horizon), device=self.device)   # Equality constraints
        self.b = torch.empty(0, device=self.device)
        self.C = torch.empty((0, self.transition_dim * self.horizon), device=self.device)   # Inequality constraints
        self.d = torch.empty(0, device=self.device)
        self.lb = torch.empty(0, device=self.device)
        self.ub = torch.empty(0, device=self.device)

        self.safety_constraints = BoxConstraints(horizon=horizon, transition_dim=transition_dim, normalizer=self.normalizer, 
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

    def __call__(self, trajectory, horizon=None, constraints=None):
        """
            trajectory: np.ndarray of shape (batch_size, horizon, transition_dim) or (horizon, transition_dim)
            Solve an optimization problem of the form 
                \hat z =   argmin_z 1/2 z^T Q z + r^T z
                        subject to  Az  = b
                                    Cz <= d
                                    
        """
        
        dims = trajectory.shape

        # Reshape the trajectory to a batch of vectors (from B x H x T to B x (HT) or a vector (from H x T to HT)
        # trajectory = trajectory.reshape(trajectory.shape[0], -1) if trajectory.ndim == 3 else trajectory.view(-1)
        if trajectory.ndim == 2:        # From H x T to HT
            batch_size = 1
            trajectory = trajectory.view(-1)
        else:      # From B x H x T to B x (HT)
            batch_size = trajectory.shape[0]
            trajectory = trajectory.view(trajectory.shape[0], -1)

        # Cost
        r = - trajectory @ self.Q

        # Constraints
        A = self.A
        b = self.b
        C = self.C
        d = self.d

        if self.skip_initial_state:
            s_0 = trajectory[:self.transition_dim] if batch_size == 1 else trajectory[0, :self.transition_dim]    # Current state, should be kept fixed
            counter = 0
            for constraint in self.dynamic_constraints.constraint_list:
                if constraint[0] == 'deriv':
                    x_idx = int(constraint[1][0])
                    dx_idx = int(constraint[1][1])
                    A[counter * (self.horizon - 1), x_idx] = 0
                    A[counter * (self.horizon - 1), dx_idx] = 0

                    if self.normalizer is not None:
                        # normalizer = self.normalizer    # for simple testing
                        normalizer = self.normalizer.normalizers['observations']
                        x_min, x_max = normalizer.mins[x_idx], normalizer.maxs[x_idx]
                        dx_min, dx_max = normalizer.mins[dx_idx], normalizer.maxs[dx_idx]
                        x_diff = x_max - x_min
                        dx_diff = dx_max - dx_min
                        dx_sum = dx_max + dx_min

                        b[counter * (self.horizon - 1)] = - x_diff * s_0[x_idx] - dx_diff * s_0[dx_idx] * self.dt - dx_sum * self.dt
                    else:
                        b[counter * (self.horizon - 1)] = -s_0[x_idx] - s_0[dx_idx] * self.dt
                    counter += 1

        # Add additional constraints (if any)
        if constraints is not None:
            for constraint in constraints:
                constraint.build_matrices()
                A = torch.cat([A, constraint.A], dim=0)
                b = torch.cat([b, constraint.b], dim=0)
                C = torch.cat([C, constraint.C], dim=0)
                d = torch.cat([d, constraint.d], dim=0)        

        # Solve optimization problem with qpth solver
        sol = QPFunction()(self.Q, r, C, d, A, b)
        sol = sol.view(dims)

        # Solve optimization problem with lqp_py solver. TODO: Implement
        # A = A.repeat(batch_size, 1, 1)
        # b = b.repeat(batch_size, 1)
        # C = C.repeat(batch_size, 1, 1)
        # d = d.repeat(batch_size, 1)
        # lb = self.lb.repeat(batch_size, 1)
        # ub = self.ub.repeat(batch_size, 1)

        return sol

    def append_constraint(self, constraint):
        self.C = torch.cat([self.C, constraint.C], dim=0)
        self.d = torch.cat([self.d, constraint.d], dim=0)
        self.A = torch.cat([self.A, constraint.A], dim=0)
        self.b = torch.cat([self.b, constraint.b], dim=0)      


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


class BoxConstraints(Constraints):

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

    # def build_matrices2(self, constraint_dict):
    #     """
    #         Input:
    #             constraint_dict: dict of constraints for each dimension
    #                 e.g. {'0': {'lb': -1, 'ub': 1}, '1': {'ub': 2}, '2': {'lb': -1}} --> 
    #                 x_0 in [-1, 1], x_1 in [-inf, 2], x_2 in [-1, inf], where x_i is the i-th dimension of the state or state-action vectors
    #         The matrices have the following shapes:
    #             C: (horizon * n_bounds, transition_dim * horizon)
    #             d: (horizon * n_bounds)
    #             lb: horizon * transition_dim
    #         C consists of n_bounds blocks of shape (horizon, transition_dim * horizon), where each block corresponds to a 
    #         constraint (ub or lb) on a specific dimension. The block has a 1 or -1 at the corresponding dimension and time step.
        
    #     """

    #     lb = -torch.inf * torch.ones(self.horizon * self.transition_dim, device=self.device)
    #     ub = torch.inf * torch.ones(self.horizon * self.transition_dim, device=self.device)

    #     for dim, bounds in constraint_dict.items():     # A dimension can have more than one constraint
    #         dim = int(dim)
    #         for bound in bounds:
    #             if self.skip_initial_state:
    #                 lb[dim + self.transition_dim::self.transition_dim] = bounds[bound] if bound == 'lb' else lb[dim + self.transition_dim::self.transition_dim]
    #                 ub[dim + self.transition_dim::self.transition_dim] = bounds[bound] if bound == 'ub' else ub[dim + self.transition_dim::self.transition_dim]
    #             else:
    #                 lb[dim::self.transition_dim] = bounds[bound] if bound == 'lb' else lb[dim::self.transition_dim]
    #                 ub[dim::self.transition_dim] = bounds[bound] if bound == 'ub' else ub[dim::self.transition_dim]
                
    #             # if self.skip_initial_state:
    #             #     h = self.horizon - 1
    #             # else:
    #             #     h = self.horizon
    #             C_append = torch.zeros(self.horizon, self.transition_dim * self.horizon, device=self.device)
    #             d_append = torch.zeros(self.horizon, device=self.device)
                
    #             sign = 1 if bound == 'ub' else -1
                
    #             for t in range(self.horizon):
    #                 # col = (t + 1) * self.transition_dim + dim if self.skip_initial_state else t * self.transition_dim + dim

    #                 C_append[t, t * self.transition_dim + dim] = sign 
    #                 d_append[t] = sign * bounds[bound]    
                
    #             if self.normalizer is not None:
    #                 x_min = self.normalizer.normalizers['observations'].mins[dim]
    #                 x_max = self.normalizer.normalizers['observations'].maxs[dim]
    #                 # x_min = self.normalizer.mins[dim]
    #                 # x_max = self.normalizer.maxs[dim]
    #                 C_append = C_append * (x_max - x_min) / 2
    #                 # d_append = d_append - sign * x_min - (x_max - x_min) / 2
    #                 d_append = d_append - sign * (x_min + x_max) / 2

    #             self.C = torch.cat((self.C, C_append), dim=0)
    #             self.d = torch.cat((self.d, d_append), dim=0)

    #     if self.skip_initial_state:
    #         mask = torch.ones(self.C.shape[0], dtype=torch.bool, device=self.device)
    #         mask[::self.horizon] = False
    #         self.C = self.C[mask]
    #         self.d = self.d[mask]

    #     self.lb = lb
    #     self.ub = ub          

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

                # if self.skip_initial_state: --> Do that in the projection method because it needs the current state. For that, record the relevant rows
                #     mat_append[0, x_idx] = 0
                #     mat_append[0, dx_idx] = 0

                self.A = torch.cat((self.A, mat_append), dim=0)
                self.b = torch.cat((self.b, vec_append), dim=0)
