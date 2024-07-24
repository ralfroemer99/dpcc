import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from diffuser.sampling import Projector
from diffuser.datasets import LimitsNormalizer

# \tau = [s_0, s_1, ..., s_T], where T+1 is the horizon length and s_i = [x_i, y_i, v_x_i, v_y_i]
# Dynamic constraints: x_{i+1} = x_i + v_x_i, y_{i+1} = y_i + v_y_i

dt = 0.1

# Generate a batch of noisy trajectories
tau_noisy = torch.tensor([
    [0.5, -0.5],
    [1.5, -1.0],
    ], device='cuda')
horizon = tau_noisy.shape[0]
transition_dim = tau_noisy.shape[1]
# tau_noisy = tau_noisy.repeat(3, 1, 1)

# Specify box constraints
constraints_specs = [
    {'ineq': ([1, 0], 2)},
    {'deriv': [0, 1]}
    ]
# constraints_specs = [{'lb': [-1, -np.inf, -np.inf], 
#                      'ub': [0, 2, np.inf],
#                      'eq': ([0, 2, 3], 2)}]

# Normalizer
normalizer = LimitsNormalizer(tau_noisy.cpu().numpy())
tau_noisy_normalized = torch.tensor(normalizer.normalize(tau_noisy.cpu().numpy()), device='cuda')

# Project normalized trajectory and unnormalize it
projector_with_normalization = Projector(horizon, transition_dim, dt=dt, constraints_specs=constraints_specs,
                                        skip_initial_state=False, normalizer=normalizer)

tau_normalized = projector_with_normalization(tau_noisy_normalized)
tau1 = normalizer.unnormalize(tau_normalized.cpu().numpy())

# Project unnormalized trajectory
projector_without_normalization = Projector(horizon, transition_dim, dt=dt, constraints_specs=constraints_specs,
                                        skip_initial_state=False, normalizer=None)

tau2 = projector_without_normalization(tau_noisy).cpu().numpy()

# Compare
print(tau1)
print(tau2)

# Plot the noisy and projected trajectories
# tau = tau.cpu().numpy()
# tau_noisy = tau_noisy.cpu().numpy()

# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# ax[0].plot(tau_noisy[:, 0], tau_noisy[:, 1], 'b')
# ax[0].plot(tau[:, 0], tau[:, 1], 'r') 

# ax[1].plot(tau_noisy[:, 2], tau_noisy[:, 3], 'b')
# ax[1].plot(tau[:, 2], tau[:, 3], 'r')

# plt.show()
