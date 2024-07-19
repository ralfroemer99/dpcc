import torch
from qpth.qp import QPFunction


class Projector:

    def __init__(self, horizon, transition_dim, constraints_specs=[], normalizer=None, dt=0.1, 
                 skip_initial_state=True, device='cuda'):
        self.horizon = horizon
        self.transition_dim = transition_dim
        self.dt = torch.tensor(dt, device=device)
        self.skip_initial_state = skip_initial_state
        self.device = device

        self.normalizer = normalizer

        if normalizer is not None:
            if normalizer.normalizers['actions'].__class__.__name__ != 'LimitsNormalizer' or \
                normalizer.normalizers['observations'].__class__.__name__ != 'LimitsNormalizer':
                raise ValueError('Normalizer not supported')

        self.Q = torch.eye(transition_dim * horizon, device=self.device)
        self.A = torch.empty((0, self.transition_dim * self.horizon), device=self.device)
        self.b = torch.empty(0, device=self.device)
        self.C = torch.empty((0, self.transition_dim * self.horizon), device=self.device)
        self.d = torch.empty(0, device=self.device)
        self.lb = torch.empty(0, device=self.device)
        self.ub = torch.empty(0, device=self.device)

        for constraint_spec in constraints_specs:
            constraint = BoxConstraints(horizon=horizon, transition_dim=transition_dim, normalizer=self.normalizer, 
                                        skip_initial_state=self.skip_initial_state, device=self.device)     # TODO: Include DynamicConstraints later
            constraint.build_matrices(constraint_spec)
            self.append_constraint(constraint)

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

        # Add additional constraints (if any)
        # if constraints is not None:
        #     A, b, C, d = [torch.cat(items, dim=0) for items in zip((A, b, C, d), *(c.build_matrices() for c in constraints))]
        A = self.A
        b = self.b
        C = self.C
        d = self.d
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
        
    def build_matrices(self, constraint_dict):
        """
            Input:
                constraint_dict: dict of constraints for each dimension
                    e.g. {'0': {'lb': -1, 'ub': 1}, '1': {'ub': 2}, '2': {'lb': -1}} --> 
                    x_0 in [-1, 1], x_1 in [-inf, 2], x_2 in [-1, inf], where x_i is the i-th dimension of the state or state-action vectors
            The matrices have the following shapes:
                C: (horizon * n_bounds, transition_dim * horizon)
                d: (horizon * n_bounds)
                lb: horizon * transition_dim
            C consists of n_bounds blocks of shape (horizon, transition_dim * horizon), where each block corresponds to a 
            constraint (ub or lb) on a specific dimension. The block has a 1 or -1 at the corresponding dimension and time step.
        
        """

        lb = -torch.inf * torch.ones(self.horizon * self.transition_dim, device=self.device)
        ub = torch.inf * torch.ones(self.horizon * self.transition_dim, device=self.device)

        for dim, bounds in constraint_dict.items():     # A dimension can have more than one constraint
            dim = int(dim)
            for bound in bounds:
                if self.skip_initial_state:
                    lb[dim + self.transition_dim::self.transition_dim] = bounds[bound] if bound == 'lb' else lb[dim + self.transition_dim::self.transition_dim]
                    ub[dim + self.transition_dim::self.transition_dim] = bounds[bound] if bound == 'ub' else ub[dim + self.transition_dim::self.transition_dim]
                else:
                    lb[dim::self.transition_dim] = bounds[bound] if bound == 'lb' else lb[dim::self.transition_dim]
                    ub[dim::self.transition_dim] = bounds[bound] if bound == 'ub' else ub[dim::self.transition_dim]
                
                # if self.skip_initial_state:
                #     h = self.horizon - 1
                # else:
                #     h = self.horizon
                C_append = torch.zeros(self.horizon, self.transition_dim * self.horizon, device=self.device)
                d_append = torch.zeros(self.horizon, device=self.device)
                
                sign = 1 if bound == 'ub' else -1
                
                for t in range(self.horizon):
                    # col = (t + 1) * self.transition_dim + dim if self.skip_initial_state else t * self.transition_dim + dim

                    C_append[t, t * self.transition_dim + dim] = sign 
                    d_append[t] = sign * bounds[bound]
                    
                # Unnormalize the constraints. TODO: Extend to actions by checking if dim <= action_dim
                # We have Cs <= d, where s is the unnormalized state vector. This is converted to C's_n <= d', where s_n is the normalized state vector,
                # by using the fact that s = (s_n + 1) * (s_max - s_min) / 2 + s_min.
                if self.normalizer is not None:
                    x_min = self.normalizer.normalizers['observations'].mins[dim]
                    x_max = self.normalizer.normalizers['observations'].maxs[dim]
                    # x_min = self.normalizer.mins[dim]
                    # x_max = self.normalizer.maxs[dim]
                    C_append = C_append * (x_max - x_min) / 2
                    # d_append = d_append - sign * x_min - (x_max - x_min) / 2
                    d_append = d_append - sign * (x_min + x_max) / 2

                self.C = torch.cat((self.C, C_append), dim=0)
                self.d = torch.cat((self.d, d_append), dim=0)

        if self.skip_initial_state:
            mask = torch.ones(self.C.shape[0], dtype=torch.bool, device=self.device)
            mask[::self.horizon] = False
            self.C = self.C[mask]
            self.d = self.d[mask]

        self.lb = lb
        self.ub = ub


class DynamicConstraints(Constraints):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_matrices(self): 
        pass        # To be implemented
