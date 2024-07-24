import minari
import numpy as np
import matplotlib.pyplot as plt

# exp = 'pointmaze-umaze-dense-v2'
exp = 'antmaze-umaze-v1'

if 'pointmaze' in exp:
    obs_indices = {'x': 0, 'y': 1, 'vx': 2, 'vy': 3, 'goal_x': 4, 'goal_y': 5}
elif 'antmaze' in exp:
    obs_indices = {'x': 0, 'y': 1, 'z':2, 'vx': 15, 'vy': 16, 'vz': 17, 'goal_x': 29, 'goal_y': 30, 'qx': 3, 'qy': 4, 'qz': 5, 'qw': 6}

dataset = minari.load_dataset(exp, download=True)
env = dataset.recover_environment(render_mode='human', eval_env=True)
env.reset()
env.render()
for _ in range(20):
    action = env.action_space.sample()
    obs, rew, terminated, truncated, info = env.step(action)
    env.render()

n_plot = 500

episodes_generator = dataset.iterate_episodes(episode_indices=np.arange(n_plot))

final_distances = np.zeros(n_plot)
episode_lengths = np.zeros(n_plot)

fig, ax1 = plt.subplots(5, 8)       # Plot episodes individually: x, y, vx, vy, ax, ay, reward, x-y-plane
fig, ax2 = plt.subplots(1, 4)       # Plot trajectories in x-y plane

for i in range(n_plot):
    episode = next(episodes_generator)
    if 'antmaze' in exp:
        observations = np.concatenate((episode.observations['achieved_goal'], episode.observations['observation'], episode.observations['desired_goal']), axis=1)
    else:
        observations = np.concatenate((obs['observation'], obs['desired_goal']), axis=1)

    # observations = np.concatenate([episode.observations[key] for key in episode.observations])
    episode_lengths[i] = observations.shape[0]
    actions = episode.actions
    rewards = episode.rewards
    terminals = episode.terminations

    ax2[0].plot(observations[:, obs_indices['x']], observations[:, obs_indices['y']])
    ax2[1].plot(observations[0, obs_indices['x']], observations[0, obs_indices['y']], 'go')       # Start
    ax2[1].plot(observations[1, obs_indices['goal_x']], observations[1, obs_indices['goal_y']], 'ro')       # Goal

    final_distances[i] = np.linalg.norm(observations[-1, [obs_indices['x'], obs_indices['y']]] - observations[-1, [obs_indices['goal_x'], obs_indices['goal_y']]])

    if i > ax1.shape[0] - 1:
        continue

    ax1[i, 0].plot(observations[:, obs_indices['x']])
    ax1[i, 1].plot(observations[:, obs_indices['y']])
    ax1[i, 2].plot(observations[:, obs_indices['vx']])
    ax1[i, 3].plot(observations[:, obs_indices['vy']])

    for j in range(2):
        ax1[i, j + 4].plot(actions[:, j])

    ax1[i, 6].plot(rewards)
    ax1[i, 7].plot(observations[:, obs_indices['x']], observations[:, obs_indices['y']])
    ax1[i, 7].plot(observations[0, obs_indices['x']], observations[0, obs_indices['y']], 'go')    # Start
    ax1[i, 7].plot(observations[1, obs_indices['goal_x']], observations[1, obs_indices['goal_y']], 'ro')    # Goal

ax1[0, 0].set_title('x')
ax1[0, 1].set_title('y')
ax1[0, 2].set_title('vx')
ax1[0, 3].set_title('vy')
ax1[0, 4].set_title('ax')
ax1[0, 5].set_title('ay')
ax1[0, 6].set_title('Reward')
for i in range(ax1.shape[1]):
    ax1[-1, i].set_xlabel('Timestep')
ax2[0].set_title('Trajectories in x-y plane')
ax2[1].set_title('Start (green) and goal (red) points')

ax2[2].plot(episode_lengths, 'ro')
ax2[3].plot(final_distances, 'ro')

print(f'Average episode length: {np.mean(episode_lengths)}')

plt.show()
