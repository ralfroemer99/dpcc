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
plt.figure(figsize=(10, 6))
for i in range(observations.shape[0]):
    x_coords = observations[i, :path_lengths[i], 0]
    y_coords = observations[i, :path_lengths[i], 1]
    plt.plot(x_coords, y_coords)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Trajectories in the X-Y Plane')
plt.legend()
plt.show()
plt.savefig('trajectories.png')

# Plot episode lengths
plt.figure(figsize=(10, 6))
plt.plot(path_lengths, marker='o', linestyle='None')
plt.xlabel('Episode Length')
plt.ylabel('Frequency')
plt.title('Distribution of Episode Lengths')
plt.show()
plt.savefig('episode_lengths.png')
