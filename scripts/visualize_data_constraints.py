import os, yaml
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import diffuser.utils as utils
from d3il.agents.utils.sim_path import sim_framework_path

# Load configuration
with open('config/projection_eval.yaml', 'r') as file:
    config = yaml.safe_load(file)

# General
exp = 'avoiding-d3il'
halfspace_variants = config['avoiding_halfspace_variants']
ax_limits = config['ax_limits'][exp]

# Constraint projection
constraint_types = config['constraint_types']

obs_indices = {'x_des': 0, 'y_des': 1, 'x': 2, 'y': 3}
obs_dim = 4
action_dim = 2

data_directory = 'environments/dataset/data/data/avoiding/data'
inputs = []
actions = []
path_lengths = []

data_dir = sim_framework_path(data_directory)
state_files = os.listdir(data_dir)

for file in state_files:
    with open(os.path.join(data_dir, file), 'rb') as f:
        env_state = pickle.load(f)

    zero_obs = np.zeros((1, 200, obs_dim), dtype=np.float32)
    zero_action = np.zeros((1, 200, action_dim), dtype=np.float32)

    # robot and box posistion
    robot_des_pos = env_state['robot']['des_c_pos'][:, :2]
    robot_c_pos = env_state['robot']['c_pos'][:, :2]

    input_state = np.concatenate((robot_des_pos, robot_c_pos), axis=-1)

    vel_state = robot_des_pos[1:] - robot_des_pos[:-1]
    valid_len = len(vel_state)

    zero_obs[0, :valid_len, :] = input_state[:-1]
    zero_action[0, :valid_len, :] = vel_state

    inputs.append(zero_obs)
    actions.append(zero_action)
    path_lengths.append(valid_len)

observations = np.concatenate(inputs)
actions = np.concatenate(actions)


## -------------- Plot all trajectories in the x-y plane -------------- ##
fig, ax = plt.subplots(figsize=(9, 10))
for i in range(observations.shape[0]):
    x_coords = observations[i, :path_lengths[i], 0]
    y_coords = observations[i, :path_lengths[i], 1]
    ax.plot(x_coords, y_coords)

utils.plot_environment_constraints(exp, ax)
ax.plot([ax_limits[0][0], ax_limits[0][1]], [0.35, 0.35], color=[0.4, 1, 0.4], linewidth=5)

# ax.set_title('Trajectories in the X-Y Plane')
ax.set_facecolor([1, 1, 0.9])
ax.set_xlim(ax_limits[0])
ax.set_ylim(ax_limits[1])
fig.savefig('figures/avoiding_data.png', bbox_inches='tight')
# plt.savefig('avoiding_data.pdf', bbox_inches='tight', format='pdf')

# ------------ Check how many trajectories satisfy the constraints ------------ #
fig, axes = plt.subplots(1, 3, figsize=(30, 10))
for j, halfspace_variant in enumerate(halfspace_variants):
    if halfspace_variant == 'top-left-hard':
        polytopic_constraints = [config['halfspace_constraints'][exp][0]]
        obstacle_constraints = [config['obstacle_constraints'][exp][3]]
    elif halfspace_variant == 'top-right-hard':
        polytopic_constraints = [config['halfspace_constraints'][exp][1]]
        obstacle_constraints = [config['obstacle_constraints'][exp][4]]
    elif halfspace_variant == 'both-hard':
        polytopic_constraints = [config['halfspace_constraints'][exp][2], config['halfspace_constraints'][exp][3]]
        obstacle_constraints = [config['obstacle_constraints'][exp][5]]
    
    feasible_indices = []
    for traj_idx in range(observations.shape[0]):
        feasible = True
        for t in range(path_lengths[traj_idx]):
            observation = observations[traj_idx, t]

            # Check polytopic constraints
            for constraint in polytopic_constraints:
                m = (constraint[1][1] - constraint[0][1]) / (constraint[1][0] - constraint[0][0])
                b = constraint[0][1] - m * constraint[0][0]
                if (constraint[2] == 'below' and observation[obs_indices['y']] > m * observation[obs_indices['x']] + b) or \
                    (constraint[2] == 'above' and observation[obs_indices['y']] < m * observation[obs_indices['x']] + b):
                    feasible = False
                    break
            if not feasible:
                break

            # Check obstacle constraints
            for constraint in obstacle_constraints:
                center = constraint['center']
                radius = constraint['radius'] + 0.025
                dimensions = [obs_indices[dim] for dim in constraint['dimensions']]
                if constraint['type'] == 'sphere_outside' and np.linalg.norm(observation[dimensions] - center) < radius:
                    feasible = False
                    break
            if not feasible:
                break
        if feasible:
            feasible_indices.append(traj_idx)

    print(f'Halfspace variant: {halfspace_variant}')
    print(f'Number of feasible trajectories: {len(feasible_indices)}/{observations.shape[0]}, {len(feasible_indices) / observations.shape[0] * 100:.2f}%')

    ax = axes[j]
    # for i in range(observations.shape[0]):
    #     x_coords = observations[i, :path_lengths[i], 0]
    #     y_coords = observations[i, :path_lengths[i], 1]
    #     ax.plot(x_coords, y_coords)

    utils.plot_environment_constraints(exp, ax)
    ax.plot([ax_limits[0][0], ax_limits[0][1]], [0.35, 0.35], color=[0.4, 1, 0.4], linewidth=5)
    
    utils.plot_halfspace_constraints(exp, polytopic_constraints, ax, ax_limits)
    for constraint in obstacle_constraints:
        ax.add_patch(matplotlib.patches.Circle(constraint['center'], constraint['radius'], color='b', alpha=0.2))
    
    ax.set_facecolor([1, 1, 0.9])
    ax.set_xlim(ax_limits[0])
    ax.set_ylim(ax_limits[1])
fig.savefig('figures/avoiding_constraints.png', bbox_inches='tight')
# fig.savefig('avoiding_constraints.pdf', bbox_inches='tight', format='pdf')

# Plot velocities
fig, ax = plt.subplots(1, 2, figsize=(10, 6))
for i in range(actions.shape[0]):
    x_vel = actions[i, :path_lengths[i], 0]
    y_vel = actions[i, :path_lengths[i], 1]
    ax[0].plot(x_vel)
    ax[1].plot(y_vel)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('X Velocity')
ax[0].set_title('X Velocities')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Y Velocity')
ax[1].set_title('Y Velocities')
fig.savefig('figures/avoiding_velocities.png', bbox_inches='tight')

# Plot episode lengths
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(path_lengths, marker='o', linestyle='None')
ax.set_xlabel('Episode')
ax.set_ylabel('Episode Length')
ax.set_title('Distribution of Episode Lengths')
fig.savefig('figures/avoiding_episode_lengths.png', bbox_inches='tight')
plt.show()
