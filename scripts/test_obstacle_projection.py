import time
from copy import copy
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from diffuser.sampling import Projector
from diffuser.datasets import LimitsNormalizer

# \tau = [s_0, s_1, ..., s_T], where T+1 is the horizon length and s_i = [x_i, y_i, v_x_i, v_y_i]
# Dynamic constraints: x_{i+1} = x_i + v_x_i, y_{i+1} = y_i + v_y_i

dt = 0.1

# Generate a batch of noisy trajectories
tau_noisy = torch.tensor([
    [-1, -1, 3, 3],
    [-0.5, -0.5, 5, 5],
    [0, 0, 5, 5],
    [0.5, 0.5, 5, 4],
    [1, 1, 5, 5],
    ], 
    device='cuda'
)
# tau_noisy = torch.tensor([
#     [0, 0, 3, 3],
#     [0.5, 0.5, 5, 5],
#     [1, 1, 5, 5],
#     [1.5, 1.5, 5, 4],
#     [2, 2, 5, 5],
#     [2.5, 2.5, 5, 5],
#     [3, 3, 5, 5],
#     [3.5, 3.5, 5, 5],
#     ], 
#     device='cuda'
# )
horizon = tau_noisy.shape[0]
transition_dim = tau_noisy.shape[1]
tau_noisy = tau_noisy.unsqueeze(0)
# tau_noisy = tau_noisy.repeat(3, 1, 1)

obstacle_constraints = [
        # ['sphere_outside', [obs_indices['x'], obs_indices['y']], [0, -0.5], 0.1],
        ['sphere_outside', [0, 1], [-0.1, -0.1], 0.2],
    ]

dynamic_constraints = [
    ('deriv', [0, 2]),
    ('deriv', [1, 3]),
]

constraint_list = copy(obstacle_constraints)
for constraint in dynamic_constraints:
    constraint_list.append(constraint)

normalizer = LimitsNormalizer(tau_noisy[0].cpu().numpy())

projector = Projector(horizon=horizon, transition_dim=transition_dim, constraint_list=constraint_list, normalizer=normalizer, diffusion_timestep_threshold=0.2, dt=dt, solver='gurobi')
projector_no_normalization = Projector(horizon=horizon, transition_dim=transition_dim, constraint_list=constraint_list, diffusion_timestep_threshold=0.2, dt=dt, solver='gurobi')

# Normalizer
tau_noisy_normalized = torch.tensor(normalizer.normalize(tau_noisy.cpu().numpy()), device='cuda')

tau_normalized = projector.project(tau_noisy_normalized)
tau = normalizer.unnormalize(tau_normalized.cpu().numpy())

tau_no_normalization = projector_no_normalization.project(tau_noisy).cpu().numpy()

# Compare
print(tau)
print(tau_no_normalization)

# Plot the noisy and projected trajectories
tau_noisy = tau_noisy.cpu().numpy()

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(tau[0][:, 0], tau[0][:, 1], 'g')
ax.plot(tau_noisy[0][:, 0], tau_noisy[0][:, 1], 'b')
ax.plot(tau_no_normalization[0][:, 0], tau_no_normalization[0][:, 1], 'r')
# ax[0].plot(tau[:, 0], tau[:, 1], 'r') 
for constraint in obstacle_constraints:
    ax.add_patch(matplotlib.patches.Circle(constraint[2], constraint[3], color='k', alpha=0.2))

plt.show()

fig.savefig('last_plot.png')
