import time
import minari
import numpy as np
import scipy as sp
import diffuser.utils as utils
import matplotlib.pyplot as plt
from diffuser.sampling import Policy
from diffusers import DDPMScheduler, DDIMScheduler
from diffuser.sampling import Projector


exps = [
    'antmaze-umaze-v1',
    # 'pointmaze-open-dense-v2',
    # 'pointmaze-umaze-dense-v2',
    # 'pointmaze-medium-dense-v2',
    # 'pointmaze-large-dense-v2'
        ]

for exp in exps:
    class Parser(utils.Parser):
        dataset: str = exp
        config: str = 'config.' + exp
    args = Parser().parse_args('plan')

    # Get model
    diffusion_experiment = utils.load_diffusion(
        args.loadbase, args.dataset, args.diffusion_loadpath,
        epoch=args.diffusion_epoch, seed=args.seed,
    )

    diffusion_losses = diffusion_experiment.losses
    diffusion = diffusion_experiment.diffusion
    dataset = diffusion_experiment.dataset
    trainer = diffusion_experiment.trainer

    # Create scheduler
    scheduler = DDIMScheduler(num_train_timesteps=diffusion.n_timesteps)   
    # scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
    # scheduler = DDIMScheduler(num_train_timesteps=trainer.noise_scheduler.config.num_train_timesteps)
    scheduler.set_timesteps(20)                         # Steps used for inference

    # Create projector
    if diffusion.__class__.__name__ == 'GaussianDiffusion':
        trajectory_dim = diffusion.transition_dim
    else:
        trajectory_dim = diffusion.observation_dim

    # constraint_specs = [{'0': {'lb': -0.5, 'ub': 0.5}, '1': {'lb': 0, 'ub': 0.5}}]     # TODO: Get constraints from config
    constraint_list = []
    
    projector = Projector(
        horizon=args.horizon,
        transition_dim=trajectory_dim,
        constraint_list=constraint_list,
        normalizer=dataset.normalizer)

    # Create policy
    policy = Policy(
        model=diffusion,
        scheduler=scheduler,
        normalizer=dataset.normalizer,
        preprocess_fns=args.preprocess_fns,
        test_ret=args.test_ret,
        projector=None,
    )    

    # Plot losses
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    utils.plot_losses(diffusion_losses, ax=ax[0], title='Diffusion losses')
    if len(exps) == 1:
        plt.show()

    dataset = minari.load_dataset(exp, download=True)
    # env = dataset.recover_environment(eval_env=True)     # Set render_mode='human' to visualize the environment
    env = dataset.recover_environment()     # Set render_mode='human' to visualize the environment

    if 'pointmaze' in exp:
        env.env.env.env.point_env.frame_skip = 2
    if 'antmaze' in exp:
        env.env.env.env.ant_env.frame_skip = 5

    # Run policy
    n_trials = 20
    n_timesteps = 500
    fig, ax = plt.subplots(min(n_trials, 10), 5)

    action_update_every = 1
    save_samples_every = 10

    # Store a few sampled trajectories
    sampled_trajectories_all = []

    if 'pointmaze' in exp:
        obs_indices = {'x': 0, 'y': 1, 'vx': 2, 'vy': 3, 'goal_x': 4, 'goal_y': 5}
    elif 'antmaze' in exp:
        obs_indices = {'x': 0, 'y': 1, 'z':2, 'vx': 15, 'vy': 16, 'vz': 17, 'goal_x': 29, 'goal_y': 30, 'qx': 3, 'qy': 4, 'qz': 5, 'qw': 6}

    n_success = 0
    n_steps = 0
    avg_time = np.zeros(n_trials)
    for i in range(n_trials):
        obs, _ = env.reset(seed=i)
        if 'antmaze' in exp:
            obs = np.concatenate((obs['achieved_goal'], obs['observation'], obs['desired_goal']))
        else:
            obs = np.concatenate((obs['observation'], obs['desired_goal']))
        obs_buffer = []
        action_buffer = []
        reward_buffer = []
        # update_counter = 0

        sampled_trajectories = []
        for _ in range(n_timesteps):
            start = time.time()
            conditions = {0: obs}
            
            # if _ % action_update_every == 0:
            action, samples = policy(conditions, batch_size=args.batch_size, horizon=args.horizon)
            # action = env.action_space.sample()
            # prev_action = action
            # update_counter = 1
            if _ % save_samples_every == 0:
                sampled_trajectories.append(samples.observations[:, :, :])
            # else:
                # action = prev_action
                # update_counter += 1

            obs, rew, terminated, truncated, info = env.step(action)

            avg_time[i] += time.time() - start

            dist_to_goal = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
            if 'antmaze' in exp:
                obs = np.concatenate((obs['achieved_goal'], obs['observation'], obs['desired_goal']))
            else:
                obs = np.concatenate((obs['observation'], obs['desired_goal']))

            # For ant robot, check if is has flipped over or reached the goal (not provided by the environment)
            if 'antmaze' in exp:
                quat = [obs[obs_indices['qx']], obs[obs_indices['qy']], obs[obs_indices['qz']], obs[obs_indices['qw']]]
                body_orientation = sp.spatial.transform.Rotation.from_quat(quat).as_matrix()
                if obs[obs_indices['z']] < 0.3:    # Ant is likely flipped over
                # if body_orientation[2, 2] > 0.5:    # Ant is flipped over --> does not work, body_orientation[0, 0] < -0.5 seems to work (no idea why)
                    terminated = True
                    print('Ant flipped over')
                    print(body_orientation)
                if dist_to_goal <= 1:
                    info['success'] = True

            obs_buffer.append(obs)
            action_buffer.append(action)
            reward_buffer.append(rew)
            if info['success']:
                n_success += 1
                n_steps += _
                print(f'Trial {i} succeeded in {_} steps')
                avg_time[i] /= _
                break
            if terminated or truncated or _ == n_timesteps - 1:
                print(f'Trial {i} terminated in {_} steps')
                avg_time[i] /= _
                break

        sampled_trajectories_all.append(sampled_trajectories)

        if i >= 10:     # Plot only the first 10 trials
            continue
        # print(f'Average computation time: {np.mean(avg_time)}')
        plot_states = ['x', 'y', 'vx', 'vy']
        for j in range(len(plot_states)):
            ax[i, j].plot(np.array(obs_buffer)[:, obs_indices[plot_states[j]]])
            ax[i, j].set_title(['x', 'y', 'vx', 'vy'][j])
        # for j in range(2):        # Plot actions
        #     ax[i, j + 4].plot(np.array(action_buffer)[:, j])
        #     ax[i, j + 4].set_title(['ax', 'ay'][j])
        # ax[i, 4].plot(reward_buffer)
        # ax[i, 4].set_title('Reward')
        ax[i, 4].plot(np.array(obs_buffer)[:, obs_indices['x']], np.array(obs_buffer)[:, obs_indices['y']], 'b')
        ax[i, 4].plot(np.array(obs_buffer)[0, obs_indices['goal_x']], np.array(obs_buffer)[0, obs_indices['goal_y']], 'ro')  # Goal
        ax[i, 4].plot(np.array(obs_buffer)[0, obs_indices['x']], np.array(obs_buffer)[0, obs_indices['y']], 'go')    # Start

    print(f'Success rate: {n_success / n_trials}')
    if n_success > 0:
        print(f'Average number of steps in successes: {n_steps / n_success}')
    print(f'Average computation time per step: {np.mean(avg_time)}')

    if len(exps) == 1:
        plt.show()     

    fig, ax = plt.subplots(1, min(n_trials, 5), figsize=(20, 5))
    for _ in range(min(n_trials, 5)):       # Iterate over trials
        for __ in range(len(sampled_trajectories_all[_])):     # Iterate over sampled trajectories
            for ___ in range(min(args.batch_size, 4)):            # Iterate over dimensions
                ax[_].plot(sampled_trajectories_all[_][__][___, :, obs_indices['x']], sampled_trajectories_all[_][__][___, :, obs_indices['y']], 'b')
                ax[_].plot(sampled_trajectories_all[_][__][___, 0, obs_indices['x']], sampled_trajectories_all[_][__][___, 0, obs_indices['y']], 'go')    # Start
                ax[_].plot(sampled_trajectories_all[_][__][___, 0, obs_indices['goal_x']], sampled_trajectories_all[_][__][___, 0, obs_indices['goal_y']], 'ro')    # Goal
    if len(exps) == 1:
        plt.show()

    env.close()