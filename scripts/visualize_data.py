import minari
import numpy as np
import matplotlib.pyplot as plt

exp = 'pointmaze-large-dense-v2'

dataset = minari.load_dataset(exp, download=True)
episodes_generator = dataset.iterate_episodes(episode_indices=np.arange(50))

fig, ax1 = plt.subplots(5, 8)       # Plot episodes individually: x, y, vx, vy, ax, ay, reward, x-y-plane
fig, ax2 = plt.subplots(1, 1)

for i in range(50):
    episode = next(episodes_generator)
    observations = np.concatenate([episode.observations[key] for key in episode.observations], axis=1)
    # observations = episode['observations']
    actions = episode.actions
    rewards = episode.rewards
    terminals = episode.terminations

    ax2.plot(observations[:32, 4], observations[:32, 5])
    if i > ax1.shape[0] - 1:
        continue

    for j in range(4):
        ax1[i, j].plot(observations[:32, j + 4])

    for j in range(2):
        ax1[i, j + 4].plot(actions[:32, j])

    ax1[i, 6].plot(rewards)
    ax1[i, 7].plot(observations[:32, 4], observations[:32, 5])
    ax1[i, 7].plot(observations[0, 0], observations[0, 1], 'ro')

ax1[0, 0].set_title('x')
ax1[0, 1].set_title('y')
ax1[0, 2].set_title('vx')
ax1[0, 3].set_title('vy')
ax1[0, 4].set_title('ax')
ax1[0, 5].set_title('ay')
ax1[0, 6].set_title('Reward')
for i in range(ax1.shape[1]):
    ax1[-1, i].set_xlabel('Timestep')
ax2.set_title('Trajectories in x-y plane')

plt.show()
