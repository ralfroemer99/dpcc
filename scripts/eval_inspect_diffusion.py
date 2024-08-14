import time
from copy import copy
import minari
import numpy as np
import scipy as sp
import diffuser.utils as utils
import matplotlib
import matplotlib.pyplot as plt
from diffuser.sampling import Policy
from diffusers import DDPMScheduler, DDIMScheduler
from diffuser.sampling import Projector


exps = [
    # 'antmaze-umaze-v1',
    # 'pointmaze-open-dense-v2',
    'pointmaze-umaze-dense-v2',
    # 'pointmaze-medium-dense-v2',
    # 'pointmaze-large-dense-v2'
        ]

projection_variants = [
    'none',
    # 'end_safe', 
    # 'full_safe', 
    # 'end_all', 
    # 'full_all',
    ]

for exp in exps:
    class Parser(utils.Parser):
        dataset: str = exp
        config: str = 'config.' + exp
    args = Parser().parse_args('plan')

    # Load model
    diffusion_experiment = utils.load_diffusion(args.loadbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch, seed=args.seed)
    diffusion_losses = diffusion_experiment.losses
    diffusion = diffusion_experiment.diffusion
    dataset = diffusion_experiment.dataset
    trainer = diffusion_experiment.trainer

    # Create scheduler
    scheduler = DDIMScheduler(num_train_timesteps=diffusion.n_timesteps)   
    scheduler.set_timesteps(20)                         # Steps used for inference

    ## Create projector
    if diffusion.__class__.__name__ == 'GaussianDiffusion':
        trajectory_dim = diffusion.transition_dim
    else:
        trajectory_dim = diffusion.observation_dim 

    if 'pointmaze' in exp:
        obs_indices = {'x': 0, 'y': 1, 'vx': 2, 'vy': 3, 'goal_x': 4, 'goal_y': 5}
        cost_dims = [obs_indices['x'], obs_indices['y'], obs_indices['vx'], obs_indices['vy']]
    elif 'antmaze' in exp:
        obs_indices = {'x': 0, 'y': 1, 'z':2, 'qx': 3, 'qy': 4, 'qz': 5, 'qw': 6, 'hip1': 7, 'ankle1': 8, 'hip2': 9, 'ankle2': 10, 
                       'hip3': 11, 'ankle3': 12, 'hip4': 13, 'ankle4': 14, 'vx': 15, 'vy': 16, 'vz': 17, 'dhip1': 21, 'dankle1': 22,
                       'dhip2': 23, 'dankle2': 24, 'dhip3': 25, 'dankle3': 26, 'dhip4': 27, 'dankle4': 28, 'goal_x': 29, 'goal_y': 30, }
        cost_dims = [obs_indices['x'], obs_indices['y'], obs_indices['z'], obs_indices['vx'], obs_indices['vy'], obs_indices['vz']]
        
    if 'pointmaze' in exp:
        safety_constraints = [
            [[0.25, -1.5], [1.5, -0.25], 'above'],
            [[1.5, 0.25], [0.25, 1.5], 'below'],
            ]
    else:
        safety_constraints = [
            [[1.5, -6], [6, -1.5], 'above'],
            [[6, 1.5], [1.5, 6], 'below'],
            ]
    
    constraint_list_safe = []
    constraint_points = safety_constraints
    for constraint in constraint_points:
        m = (constraint[1][1] - constraint[0][1]) / (constraint[1][0] - constraint[0][0])
        d = constraint[0][1] - m * constraint[0][0]
        C_row = np.zeros(trajectory_dim)
        if constraint[2] == 'below':
            C_row[obs_indices['x']] = -m
            C_row[obs_indices['y']] = 1
        elif constraint[2] == 'above':
            C_row[obs_indices['x']] = m
            C_row[obs_indices['y']] = -1
            d *= -1
        constraint_list_safe.append(('ineq', (C_row, d)))

    constraint_list_safe_dyn = copy(constraint_list_safe)   
    if 'pointmaze' in exp:
        dynamic_constraints = [
            ('deriv', ['x', 'vx']),
            ('deriv', ['y', 'vy']),
        ]
    elif 'antmaze' in exp:
        dynamic_constraints = [
            ('deriv', ['x', 'vx']),
            ('deriv', ['y', 'vy']),
            ('deriv', ['z', 'vz']),
            # ('deriv', ['hip1', 'dhip1']),
            # ('deriv', ['ankle1', 'dankle1']),
            # ('deriv', ['hip2', 'dhip2']),
            # ('deriv', ['ankle2', 'dankle2']),
            # ('deriv', ['hip3', 'dhip3']),
            # ('deriv', ['ankle3', 'dankle3']),
            # ('deriv', ['hip4', 'dhip4']),
            # ('deriv', ['ankle4', 'dankle4']),
        ]

    for constraint in dynamic_constraints:
        constraint = ('deriv', [obs_indices[constraint[1][0]], obs_indices[constraint[1][1]]])
        constraint_list_safe_dyn.append(constraint)

    # Load dataset and create environment
    minari_dataset = minari.load_dataset(exp, download=True)
    # env = dataset.recover_environment(eval_env=True)     # Set render_mode='human' to visualize the environment
    env = minari_dataset.recover_environment(eval_env=True) if 'pointmaze' in exp else minari_dataset.recover_environment()     # Set render_mode='human' to visualize the environment

    if 'pointmaze' in exp:
        env.env.env.env.point_env.frame_skip = 2
        dt = 0.02
    elif 'antmaze' in exp:
        dt = 0.05

    ax_limits = [-1.5, 1.5] if 'pointmaze' in exp else [-6, 6]
    for variant_idx, variant in enumerate(projection_variants):
        print(f'------------------------Running {exp} - {variant}----------------------------')

        if variant == 'none':
            projector = None
        else:
            constraint_list = constraint_list_safe if 'safe' in variant else constraint_list_safe_dyn
            diffusion_timestep_threshold = 0.25 if 'full' in variant else 0
            dt = 0.02 if 'pointmaze' in exp else 0.05
            projector = Projector(
                horizon=args.horizon, 
                transition_dim=trajectory_dim, 
                constraint_list=constraint_list, 
                normalizer=dataset.normalizer, 
                diffusion_timestep_threshold=diffusion_timestep_threshold,
                dt=dt,
                cost_dims=cost_dims,
            )

        # Create policy
        policy = Policy(
            model=diffusion,
            scheduler=scheduler,
            normalizer=dataset.normalizer,
            preprocess_fns=args.preprocess_fns,
            test_ret=args.test_ret,
            projector=projector,
            # **sample_kwargs
            return_diffusion=True,
        )   

        # Run policy
        if 'pointmaze' in exp:
            # seeds = [7, 10, 11, 16, 24, 28, 31, 33, 39, 41, 43, 44, 45, 46, 48]     # Good seeds for pointmaze-umaze-dense-v2: [7, 10, 11, 16, 24, 28, 31, 33, 39, 41, 43, 44, 45, 46, 48]
            seeds = [7, 10]
        else:
            seeds = [0, 1, 2, 3, 4, 5, 6, 7] 
        n_trials = 2
        n_timesteps = 100 if 'pointmaze' in exp else 300
        fig, ax = plt.subplots(min(n_trials, 5), 5)

        action_update_every = 1
        save_samples_every = 10 if 'pointmaze' in exp else 50

        # Store a few sampled trajectories
        sampled_trajectories_all = []

        for i in range(n_trials):
            obs, _ = env.reset(seed=seeds[i])
            if 'antmaze' in exp:
                obs = np.concatenate((obs['achieved_goal'], obs['observation'], obs['desired_goal']))
            else:
                obs = np.concatenate((obs['observation'], obs['desired_goal']))
            obs_buffer = []
            action_buffer = []

            sampled_trajectories = []
            for trial_idx in range(n_timesteps):
                start = time.time()
                conditions = {0: obs}
                
                action, samples = policy(conditions, batch_size=args.batch_size, horizon=args.horizon)

                if trial_idx % save_samples_every == 0:
                    sampled_trajectories.append(samples.observations)       # Shape of samples.observations is (batch_size, T, horizon, observation_dim)

                obs, rew, terminated, truncated, info = env.step(action)

                dist_to_goal = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
                if 'antmaze' in exp:
                    obs = np.concatenate((obs['achieved_goal'], obs['observation'], obs['desired_goal']))
                else:
                    obs = np.concatenate((obs['observation'], obs['desired_goal']))

                # For ant robot, check if is has flipped over or reached the goal (not provided by the environment)
                if 'antmaze' in exp:
                    if obs[obs_indices['z']] < 0.3:    # Ant is likely flipped over
                    # if body_orientation[2, 2] > 0.5:    # Ant is flipped over --> does not work, body_orientation[0, 0] < -0.5 seems to work (no idea why)
                        terminated = True
                    if dist_to_goal <= 1:
                        info['success'] = True

                obs_buffer.append(obs)
                action_buffer.append(action)
                if info['success']:
                    print(f'Trial {i} succeeded in {trial_idx} steps')
                    break
                if terminated or truncated or trial_idx == n_timesteps - 1:
                    print(f'Trial {i} terminated in {trial_idx} steps')
                    break

            sampled_trajectories_all.append(sampled_trajectories)

            if i >= 5:     # Plot only the first 5 trials
                continue
            # print(f'Average computation time: {np.mean(avg_time)}')
            plot_states = ['x', 'y', 'vx', 'vy']
            for j in range(len(plot_states)):
                ax[i, j].plot(np.array(obs_buffer)[:, obs_indices[plot_states[j]]])
                ax[i, j].set_title(['x', 'y', 'vx', 'vy'][j])
            ax[i, 4].plot(np.array(obs_buffer)[:, obs_indices['x']], np.array(obs_buffer)[:, obs_indices['y']], 'b')
            ax[i, 4].plot(np.array(obs_buffer)[0, obs_indices['goal_x']], np.array(obs_buffer)[0, obs_indices['goal_y']], 'ro')  # Goal
            ax[i, 4].plot(np.array(obs_buffer)[0, obs_indices['x']], np.array(obs_buffer)[0, obs_indices['y']], 'go')    # Start
        
        # fig.savefig(f'logs/last_plot.png')
        # if len(exps) == 1:
        #     plt.show()     

        # -----------------------------------------------------------------------------#
        # ---------------- Visualize denoising process of trajectories ----------------#
        # -----------------------------------------------------------------------------#
        show_which = [20, 15, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        fig, ax = plt.subplots(min(n_trials, 5), len(show_which), figsize=(40, 15))
        for t in range(len(show_which)):
            t_show = show_which[t]
            for trial_idx in range(min(n_trials, 5)):       # Iterate over trials
                for __ in range(len(sampled_trajectories_all[trial_idx])):     # Iterate over sampled trajectories
                    for ___ in range(min(args.batch_size, 4)):            # Iterate over dimensions
                        ax[trial_idx, t].plot(sampled_trajectories_all[trial_idx][__][___, diffusion.n_timesteps - t_show, :, obs_indices['x']], 
                                            sampled_trajectories_all[trial_idx][__][___, diffusion.n_timesteps - t_show, :, obs_indices['y']], 'b')
                        ax[trial_idx, t].plot(sampled_trajectories_all[trial_idx][__][___, diffusion.n_timesteps - t_show, 0, obs_indices['x']], 
                                            sampled_trajectories_all[trial_idx][__][___, diffusion.n_timesteps - t_show, 0, obs_indices['y']], 'go')    # Start
                        ax[trial_idx, t].plot(sampled_trajectories_all[trial_idx][__][___, diffusion.n_timesteps - t_show, 0, obs_indices['goal_x']], 
                                            sampled_trajectories_all[trial_idx][__][___, diffusion.n_timesteps - t_show, 0, obs_indices['goal_y']], 'ro')    # Goal
                ax[trial_idx, t].set_xlim(ax_limits)
                ax[trial_idx, t].set_ylim(ax_limits)

                # Plot constraints
                if 'pointmaze' in exp:
                    ax[trial_idx, t].add_patch(matplotlib.patches.Rectangle((-1.5, -0.5), 2, 1, color='k', alpha=0.2))
                else:
                    ax[trial_idx, t].add_patch(matplotlib.patches.Rectangle((-6, -2), 8, 4, color='k', alpha=0.2))
                
                # if variant is not 'none':
                for constraint in constraint_points:
                    mat = np.zeros((3, 2))
                    mat[:2] = constraint[:2]
                    if 'pointmaze' in exp:
                        mat[2] = np.array([1.5, -1.5]) if constraint[2] == 'above' else np.array([1.5, 1.5])
                    elif 'antmaze' in exp:
                        mat[2] = np.array([6, -6]) if constraint[2] == 'above' else np.array([6, 6])
                    ax[trial_idx, t].add_patch(matplotlib.patches.Polygon(mat, color='c', alpha=0.2))

            ax[0, t].set_title(f'Diff. time={t_show}')
        
        # fig.savefig(f'{args.savepath}/inspect_diffusion_denoising_{variant}.png') 
        
        # -----------------------------------------------------------------------------#
        # ---------------- Visualize derivative error of trajectories -----------------#
        # -----------------------------------------------------------------------------#
        derivative_errors = np.zeros((len(dynamic_constraints), diffusion.n_timesteps + 1))
        fig, ax = plt.subplots(diffusion.n_timesteps + 1, min(n_trials, 5), figsize=(20, 20))
        for t_show in reversed(range(diffusion.n_timesteps + 1)):
            for dim_idx, constraint in enumerate(dynamic_constraints):
                error_curr = 0
                counter = 0
                for trial_idx in range(min(n_trials, 5)):       # Iterate over trials
                    for __ in range(len(sampled_trajectories_all[trial_idx]) - 1):     # Iterate over sampled trajectories
                        x0 = sampled_trajectories_all[trial_idx][__][:, t_show, :-1, obs_indices[constraint[1][0]]]
                        v0 = sampled_trajectories_all[trial_idx][__][:, t_show, :-1, obs_indices[constraint[1][1]]]
                        x1 = sampled_trajectories_all[trial_idx][__][:, t_show, 1:, obs_indices[constraint[1][0]]]
                        error_curr += ((x1 - x0 - v0 * dt)**2).mean()
                        counter += 1
                derivative_errors[dim_idx, t_show] = error_curr / counter
        fig, ax = plt.subplots(len(dynamic_constraints), 1, figsize=(10, 20))
        for dim_idx, constraint in enumerate(dynamic_constraints):
            ax[dim_idx].plot(derivative_errors[dim_idx])
            ax[dim_idx].set_title(f'Error for derivative relationship {constraint[1][0]}-{constraint[1][1]} with dt={dt}')
            ax[dim_idx].set_xlabel('Diffusion time')

        # fig.savefig(f'{args.savepath}/inspect_diffusion_derivative_error_{variant}.png')

        # if len(exps) == 1:
        plt.show()

        env.close()