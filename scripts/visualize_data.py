import minari
import numpy as np
import matplotlib.pyplot as plt

# exp = 'pointmaze-umaze-dense-v2'
exp = 'antmaze-umaze-v1'

if 'pointmaze' in exp:
            obs_indices = {'x': 0, 'y': 1, 'vx': 2, 'vy': 3, 'goal_x': 4, 'goal_y': 5}
elif 'antmaze' in exp:
    obs_indices = {'x': 0, 'y': 1, 'z':2, 'qx': 3, 'qy': 4, 'qz': 5, 'qw': 6, 'hip1': 7, 'ankle1': 8, 'hip2': 9, 'ankle2': 10, 
                   'hip3': 11, 'ankle3': 12, 'hip4': 13, 'ankle4': 14, 'vx': 15, 'vy': 16, 'vz': 17, 'dhip1': 21, 'dankle1': 22,
                   'dhip2': 23, 'dankle2': 24, 'dhip3': 25, 'dankle3': 26, 'dhip4': 27, 'dankle4': 28, 'goal_x': 29, 'goal_y': 30, }

if 'pointmaze' in exp:
    dynamic_constraints = [
        ('deriv', [obs_indices['x'], obs_indices['vx']]),
        ('deriv', [obs_indices['y'], obs_indices['vy']]),
    ]
elif 'antmaze' in exp:
    dynamic_constraints = [
        ('deriv', [obs_indices['x'], obs_indices['vx']]),
        ('deriv', [obs_indices['y'], obs_indices['vy']]),
        ('deriv', [obs_indices['z'], obs_indices['vz']]),
        ('deriv', [obs_indices['hip1'], obs_indices['dhip1']]),
        ('deriv', [obs_indices['ankle1'], obs_indices['dankle1']]),
        ('deriv', [obs_indices['hip2'], obs_indices['dhip2']]),
        ('deriv', [obs_indices['ankle2'], obs_indices['dankle2']]),
        ('deriv', [obs_indices['hip3'], obs_indices['dhip3']]),
        ('deriv', [obs_indices['ankle3'], obs_indices['dankle3']]),
        ('deriv', [obs_indices['hip4'], obs_indices['dhip4']]),
        ('deriv', [obs_indices['ankle4'], obs_indices['dankle4']]),
    ]


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

# Get violation
# if 'pointmaze' in exp:
#     dt = env.env.env.env.point_env.frame_skip * 0.01
# if 'antmaze' in exp:
#     dt = env.env.env.env.ant_env.frame_skip * 0.01
# episode = next(episodes_generator)
# if 'antmaze' in exp:
#     observations = np.concatenate((episode.observations['achieved_goal'], episode.observations['observation'], episode.observations['desired_goal']), axis=1)
# else:
#     observations = np.concatenate((episode.observations['observation'], episode.observations['desired_goal']), axis=1)

# for constraint in dynamic_constraints:
#     constraint_type, indices = constraint
#     derivative_error = observations[1:, indices[0]] - observations[:-1, indices[0]] - observations[:-1, indices[1]] * dt
#     print(f'{constraint_type} error for dimensions {indices} with dt={dt}: {np.sqrt((derivative_error**2).mean())}')
#     numerical_dt = np.median(np.abs((observations[2:, indices[0]] - observations[1:-1, indices[0]]) / observations[1:-1, indices[1]]))
#     print(f'Numerical dt for dimensions {indices}: {numerical_dt.mean()}')

final_distances = np.zeros(n_plot)
episode_lengths = np.zeros(n_plot)
successful_episodes = np.zeros(n_plot)
fig, ax1 = plt.subplots(5, 8)       # Plot episodes individually: x, y, vx, vy, ax, ay, reward, x-y-plane
fig, ax2 = plt.subplots(1, 4)       # Plot trajectories in x-y plane

for i in range(n_plot):
    episode = next(episodes_generator)
    if 'antmaze' in exp:
        observations = np.concatenate((episode.observations['achieved_goal'], episode.observations['observation'], episode.observations['desired_goal']), axis=1)
    else:
        observations = np.concatenate((episode.observations['observation'], episode.observations['desired_goal']), axis=1)

    # observations = np.concatenate([episode.observations[key] for key in episode.observations])
    actions = episode.actions
    rewards = episode.rewards
    terminals = episode.terminations
    if 'pointmaze' in exp:
        successful_episodes[i] = 1
        episode_lengths[i] = observations.shape[0]
    else:
        successful_episodes[i] = 1 if rewards.sum() > 0 else 0            
        episode_lengths[i] = rewards[rewards == 0].shape[0] + 1


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

print(f'Average episode length: {np.mean(episode_lengths[successful_episodes == 1])}')
print(f'Maximum episode length: {np.max(episode_lengths)}')
print(f'Successful episodes: {successful_episodes.sum()}/{n_plot}')

plt.show()
