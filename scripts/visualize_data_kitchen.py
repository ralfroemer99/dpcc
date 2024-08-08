import minari
import numpy as np
import matplotlib.pyplot as plt

exp = 'kitchen-mixed-v1'

obs_indices = {'q1': 0, 'q2': 1, 'q3': 2, 'q4': 3, 'q5': 4, 'q6': 5, 'q7': 6, 'gripper_r': 7, 'gripper_l': 8,
               'dq1': 9, 'dq2': 10, 'dq3': 11, 'dq4': 12, 'dq5': 13, 'dq6': 14, 'dq7': 15, 'dgripper_r': 16, 'dgripper_l': 17,
               'q_microwave': 31, 'dq_microwave': 52,
               'kettle_x': 32, 'kettle_y': 33, 'kettle_z': 34, 'dkettle_x': 53, 'dkettle_y': 54, 'dkettle_z': 55,}
if exp == 'kitchen-mixed-v1':
    tasks = ['bottom burner', 'kettle', 'light switch', 'microwave']
elif exp == 'kitchen-complete-v1':
    tasks = ['microwave', 'kettle', 'light switch', 'slide cabinet']

dataset = minari.load_dataset(exp, download=True)
env = dataset.recover_environment(render_mode='human', eval_env=True)

# env.reset()
# env.render()
# for _ in range(20):
#     action = env.action_space.sample()
#     obs, rew, terminated, truncated, info = env.step(action)
#     env.render()

n_plot = 100

episodes_generator = dataset.iterate_episodes(episode_indices=np.arange(n_plot))

# Plot x-y-z coordinates of the episodes
coords_to_plot = ['q_microwave', 'kettle_x', 'kettle_y', 'kettle_z']
fig1, ax1 = plt.subplots(2, len(coords_to_plot), figsize=(20, 10)) 
fig2, ax2 = plt.subplots(1, len(tasks), figsize=(20, 5))

for i in range(n_plot):
    episode = next(episodes_generator)
    # print('Number of desired goals is', len(episode.observations['desired_goal']))
    # print(episode.observations['desired_goal'].keys())
    observations = episode.observations['observation']
    
    distances_to_goal = np.zeros((4, observations.shape[0]))
    for coord in coords_to_plot:
        ax1[0, coords_to_plot.index(coord)].plot(observations[:, obs_indices[coord]], label=coord)
        ax1[1, coords_to_plot.index(coord)].plot(observations[:, obs_indices['d'+coord]], label='d'+coord)
        ax1[0, coords_to_plot.index(coord)].set_title(coord)
        ax1[1, coords_to_plot.index(coord)].set_title('d'+coord)
    
    for task in tasks:
        distances_to_goal[tasks.index(task), :] = np.linalg.norm(
            episode.observations['desired_goal'][task] - episode.observations['achieved_goal'][task], axis=1)
        ax2[tasks.index(task)].plot(distances_to_goal[tasks.index(task), :], label=task)
        ax2[tasks.index(task)].set_title(task)
    
plt.show()
