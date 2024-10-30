import os
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from agents.utils.sim_path import sim_framework_path
from d3il.environments.d3il.envs.gym_avoiding_env.gym_avoiding.envs.avoiding import ObstacleAvoidanceEnv

# env = ObstacleAvoidanceEnv()

# env.start()

obs_indices = {'x_des', 'y_des', 'x', 'y'}
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

# Plot all trajectories in the x-y plane
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(observations.shape[0]):
    x_coords = observations[i, :path_lengths[i], 0]
    y_coords = observations[i, :path_lengths[i], 1]
    ax.plot(x_coords, y_coords)
centers = [[0.5, -0.1], [0.425, 0.08], [0.575, 0.08], [0.35, 0.26], [0.5, 0.26], [0.65, 0.26]]
for center in centers:
    ax.add_patch(matplotlib.patches.Circle(center, 0.025, color='k', alpha=0.2))

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Trajectories in the X-Y Plane')
plt.show()
plt.savefig('trajectories.png')

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
plt.show()
plt.savefig('velocities.png')

# Plot episode lengths
plt.figure(figsize=(10, 6))
plt.plot(path_lengths, marker='o', linestyle='None')
plt.xlabel('Episode Length')
plt.ylabel('Frequency')
plt.title('Distribution of Episode Lengths')
plt.show()
plt.savefig('episode_lengths.png')
