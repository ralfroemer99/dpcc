import minari
import numpy as np
import matplotlib.pyplot as plt

exp = 'relocate-cloned-v2'

obs_indices = {'arm_x': 0, 'arm_y': 1, 'arm_z': 2, 
               'palm_ball_rel_x': 30, 'palm_ball_rel_y': 31, 'palm_ball_rel_z': 32,
               'palm_target_rel_x': 33, 'palm_target_rel_y': 34, 'palm_target_rel_z': 35,
               'ball_target_rel_x': 36, 'ball_target_rel_y': 37, 'ball_target_rel_z': 38,}

dataset = minari.load_dataset(exp, download=True)
env = dataset.recover_environment(render_mode='human', eval_env=True)

env.reset()
env.render()
for _ in range(20):
    action = env.action_space.sample()
    obs, rew, terminated, truncated, info = env.step(action)
    env.render()

n_plot = 10

episodes_generator = dataset.iterate_episodes(episode_indices=np.arange(n_plot))

# Plot x-y-z coordinates of the episodes
# figxyz, axxyz = plt.subplots(1, 3)
fig, ax = plt.subplots(4, 3, figsize=(20, 10)) 

for i in range(n_plot):
    episode = next(episodes_generator)
    observations = episode.observations
    coords = ['x', 'y', 'z']
    for k in range(len(coords)):
        ax[0, k].plot(observations[:, obs_indices['arm_'+coords[k]]], label='arm_'+coords[k])
        ax[1, k].plot(observations[:, obs_indices['palm_ball_rel_'+coords[k]]], label='palm_ball_rel_'+coords[k])
        ax[2, k].plot(observations[:, obs_indices['palm_target_rel_'+coords[k]]], label='palm_target_rel_'+coords[k])
        ax[3, k].plot(observations[:, obs_indices['ball_target_rel_'+coords[k]]], label='ball_target_rel_'+coords[k])
        ax[0, k].set_title('arm_'+coords[k])
        ax[1, k].set_title('palm_ball_rel_'+coords[k])
        ax[2, k].set_title('palm_target_rel_'+coords[k])
        ax[3, k].set_title('ball_target_rel_'+coords[k])
    
plt.show()
