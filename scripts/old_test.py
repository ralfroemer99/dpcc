import time
import minari
import numpy as np
import diffuser.utils as utils
import diffuser.sampling as sampling
import matplotlib.pyplot as plt

exp = 'pointmaze-umaze-dense-v2'

class Parser(utils.Parser):
    dataset: str = exp
    config: str = 'config.' + exp

args = Parser().parse_args('plan')

#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#
 
diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed,
)

value_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.value_loadpath,
    epoch=args.value_epoch, seed=args.seed,
)

diffusion_losses = diffusion_experiment.losses
value_losses = value_experiment.losses

# Plot losses
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# utils.plot_losses(diffusion_losses, ax=ax[0], title='Diffusion losses')
# utils.plot_losses(value_losses, ax=ax[1], title='Value losses')
# plt.show()
# utils.check_compatibility(diffusion_experiment, value_experiment)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
value_function = value_experiment.ema

guide_config = utils.Config(args.guide, model=value_function, verbose=False)
guide = guide_config()

policy_config = utils.Config(
    args.policy,
    guide=guide,                                    # guide = None        
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    ## sampling kwargs
    scale=args.scale,                               # comment
    sample_fn=sampling.n_step_guided_p_sample,      # comment
    n_guide_steps=args.n_guide_steps,               # comment
    t_stopgrad=args.t_stopgrad,                     # comment
    scale_grad_by_std=args.scale_grad_by_std,       # comment
    verbose=False,
)
policy = policy_config()

#-----------------------------------------------------------------------------#
#----------------------------- Make environment ------------------------------#
#-----------------------------------------------------------------------------#

dataset = minari.load_dataset(exp, download=True)
env = dataset.recover_environment(render_mode='human', eval_env=True)

#-----------------------------------------------------------------------------#
#-------------------------------- Experiment ---------------------------------#
#-----------------------------------------------------------------------------#

# Plot trajectories for 10 random initial conditions
# fig, ax = plt.subplots(1, 10, figsize=(5, 5))
# episodes_generator = dataset.iterate_episodes(episode_indices=np.arange(50))
# for i in range(10):
#     episode = next(episodes_generator)
#     observations = np.concatenate((episode.observations['observation'], episode.observations['desired_goal']), axis=1)
#     conditions = {0: observations[0]}
#     _, samples = policy(conditions, batch_size=args.batch_size, verbose=False)
#     ax[i].plot(observations[:args.horizon, 0], observations[:args.horizon, 1], 'r')
#     for _ in range(5):
#         ax[i].plot(samples.observations[_, :, 0], samples.observations[_, :, 1], 'b')
# plt.show()

# Do some trials
n_trials = 5
n_timesteps = 200
fig, ax = plt.subplots(n_trials, 8)

positions = []
for i in range(n_trials):
    obs, _ = env.reset()
    obs_buffer = []
    action_buffer = []
    reward_buffer = []
    avg_time = np.zeros(n_timesteps)
    for _ in range(n_timesteps):
        # action = env.action_space.sample()
        conditions = {0: np.concatenate((obs['observation'], obs['desired_goal']))}
        start = time.time()
        action, samples = policy(conditions, batch_size=args.batch_size, verbose=False)
        avg_time[_] = time.time() - start
        obs, rew, terminated, truncated, info = env.step(action)
        obs_buffer.append(np.concatenate((obs['observation'], obs['desired_goal'])))
        action_buffer.append(action)
        reward_buffer.append(rew)
        env.render()
        if terminated or truncated:
            break
    print(f'Average computation time: {np.mean(avg_time)}')
    for j in range(4):
        ax[i, j].plot(np.array(obs_buffer)[:, j])
        ax[i, j].set_title(['x', 'y', 'vx', 'vy'][j])
    for j in range(2):
        ax[i, j + 4].plot(np.array(action_buffer)[:, j])
        ax[i, j + 4].set_title(['ax', 'ay'][j])
    ax[i, 6].plot(reward_buffer)
    ax[i, 6].set_title('Reward')
    ax[i, 7].plot(np.array(obs_buffer)[:, 0], np.array(obs_buffer)[:, 1], 'b')
    ax[i, 7].plot(np.array(obs_buffer)[0, -2], np.array(obs_buffer)[0, -1], 'ro')  # Goal
    ax[i, 7].plot(np.array(obs_buffer)[0, 0], np.array(obs_buffer)[0, 1], 'go')    # Start
    # ax[i].set_xlim([-5, 5])
    # ax[i].set_ylim([-5, 5])

plt.show()
