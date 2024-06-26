import minari
import numpy as np
import matplotlib.pyplot as plt

exp = 'pointmaze-umaze-dense-v2'
# exp = 'antmaze-umaze-diverse-v1'

dataset = minari.load_dataset(exp, download=True)
env = dataset.recover_environment(render_mode='human', eval_env=True)
episodes_generator = dataset.iterate_episodes(episode_indices=np.arange(200))

final_distances = np.zeros(200)

fig, ax1 = plt.subplots(5, 8)       # Plot episodes individually: x, y, vx, vy, ax, ay, reward, x-y-plane
fig, ax2 = plt.subplots(1, 3)       # Plot trajectories in x-y plane

for i in range(200):
    episode = next(episodes_generator)
    observations = np.concatenate([episode.observations[key] for key in episode.observations], axis=1)
    # observations = episode['observations']
    actions = episode.actions
    rewards = episode.rewards
    terminals = episode.terminations

    ax2[0].plot(observations[:, 4], observations[:, 5])
    ax2[1].plot(observations[0, 4], observations[0, 5], 'go')       # Start
    ax2[1].plot(observations[1, 2], observations[1, 3], 'ro')       # Goal

    final_distances[i] = np.linalg.norm(observations[-1, 4:6] - observations[-1, 2:4])

    if i > ax1.shape[0] - 1:
        continue

    for j in range(4):
        ax1[i, j].plot(observations[:, j + 4])

    for j in range(2):
        ax1[i, j + 4].plot(actions[:, j])

    ax1[i, 6].plot(rewards)
    ax1[i, 7].plot(observations[:, 4], observations[:, 5])
    ax1[i, 7].plot(observations[0, 4], observations[0, 5], 'go')    # Start
    ax1[i, 7].plot(observations[1, 2], observations[1, 3], 'ro')    # Goal

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
ax2[1].set_title('Start and goal points')

ax2[2].plot(final_distances, 'ro')


plt.show()
