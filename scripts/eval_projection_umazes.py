import time
from copy import copy
import minari
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import diffuser.utils as utils
from diffuser.sampling import Policy
from diffusers import DDPMScheduler, DDIMScheduler
from diffuser.sampling import Projector


exps = [
    # 'pointmaze-umaze-dense-v2'
    'antmaze-umaze-v1',
    ]

projection_variants = [
    # 'none',
    # 'end_obs', 
    # 'full_obs', 
    'end_all', 
    'full_all',
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
    scheduler.set_timesteps(20)                         # Steps used for inference

    # Create projector
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
            # [[1, -6], [6, -1], 'above'],
            # [[6, 1], [1, 6], 'below'],
            [[1.5, -6], [6, -1.5], 'above'],
            [[6, 1.5], [1.5, 6], 'below'],
            ]
    
    constraint_list_obs = []
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
        constraint_list_obs.append(('ineq', (C_row, d)))

    constraint_list_obs_dyn = copy(constraint_list_obs)   
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
            # ('deriv', [obs_indices['hip1'], obs_indices['dhip1']]),
            # ('deriv', [obs_indices['ankle1'], obs_indices['dankle1']]),
            # ('deriv', [obs_indices['hip2'], obs_indices['dhip2']]),
            # ('deriv', [obs_indices['ankle2'], obs_indices['dankle2']]),
            # ('deriv', [obs_indices['hip3'], obs_indices['dhip3']]),
            # ('deriv', [obs_indices['ankle3'], obs_indices['dankle3']]),
            # ('deriv', [obs_indices['hip4'], obs_indices['dhip4']]),
            # ('deriv', [obs_indices['ankle4'], obs_indices['dankle4']]),
        ]

    for constraint in dynamic_constraints:
        constraint_list_obs_dyn.append(constraint)

    seeds = [7, 10] if 'pointmaze' in exp else [0, 1, 2, 3, 4, 5, 6, 7]         # Good seeds for pointmaze-umaze-dense-v2: [7, 10, 11, 16, 24, 28]
    n_trials = max(2, len(seeds))
    n_timesteps = 100 if 'pointmaze' in exp else 300

    fig_all, ax_all = plt.subplots(n_trials, len(projection_variants), figsize=(20, 10))
    ax_limits = [-1.5, 1.5] if 'pointmaze' in exp else [-6, 6]

    for variant_idx, variant in enumerate(projection_variants):
        print(f'------------------------Running {exp} - {variant}----------------------------')

        minari_dataset = minari.load_dataset(exp, download=True)
        env = minari_dataset.recover_environment(eval_env=True) if 'pointmaze' in exp else minari_dataset.recover_environment()    # Set render_mode='human' to visualize the environment

        if variant == 'none':
            projector = None
        else:
            constraint_list = constraint_list_obs if 'obs' in variant else constraint_list_obs_dyn
            only_last = True if 'end' in variant else False
            dt = 0.02 if 'pointmaze' in exp else 0.05
            projector = Projector(
                horizon=args.horizon, 
                transition_dim=trajectory_dim, 
                constraint_list=constraint_list, 
                normalizer=dataset.normalizer, 
                only_last=only_last, 
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
        )    

        if 'pointmaze' in exp:
            env.env.env.env.point_env.frame_skip = 2
        if 'antmaze' in exp:
            env.env.env.env.ant_env.frame_skip = 5

        # Run policy
        fig, ax = plt.subplots(min(n_trials, 10), 6, figsize=(20, 10))
        fig.suptitle(f'{exp} - {variant}')

        action_update_every = 1
        save_samples_every = 10

        # Store a few sampled trajectories
        sampled_trajectories_all = []

        n_success = 0
        n_steps = 0
        avg_time = np.zeros(n_trials)
        for i in range(n_trials):
            seed = seeds[i] if ('pointmaze-umaze' in exp) else i
            obs, _ = env.reset(seed=seed)
            if 'antmaze' in exp:
                obs = np.concatenate((obs['achieved_goal'], obs['observation'], obs['desired_goal']))
            else:
                obs = np.concatenate((obs['observation'], obs['desired_goal']))
            obs_buffer = []
            action_buffer = []

            sampled_trajectories = []
            disable_projection = True
            for _ in range(n_timesteps):
                start = time.time()
                conditions = {0: obs}
                
                action, samples = policy(conditions, batch_size=args.batch_size, horizon=args.horizon, disable_projection=disable_projection)

                # Check whether one of the sampled trajectories violates a 
                disable_projection = True
                for constraint in constraint_list_obs:
                    c, d = constraint[1]
                    if np.any(samples.observations @ c >= d - 1e-2):   # (Close to) Violation of constraint
                        disable_projection = False
                        # print('Enabled projection at timestep', _)
                        break

                if _ % save_samples_every == 0:
                    sampled_trajectories.append(samples.observations[:, :, :])

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
                    if obs[obs_indices['z']] < 0.3:    # Ant is likely flipped over
                        terminated = True
                        print('Ant flipped over')
                    if dist_to_goal <= 1:
                        info['success'] = True

                obs_buffer.append(obs)
                action_buffer.append(action)
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
            plot_states = ['x', 'y', 'vx', 'vy']

            for j in range(len(plot_states)):
                ax[i, j].plot(np.array(obs_buffer)[:, obs_indices[plot_states[j]]])
                ax[i, j].set_title(['x', 'y', 'vx', 'vy'][j])
            
            axes = [ax[i, 4], ax_all[i, variant_idx]]
            for curr_ax in axes:
                curr_ax.plot(np.array(obs_buffer)[:, obs_indices['x']], np.array(obs_buffer)[:, obs_indices['y']], 'k')
                curr_ax.plot(np.array(obs_buffer)[0, obs_indices['x']], np.array(obs_buffer)[0, obs_indices['y']], 'go', label='Start')            # Start
                curr_ax.plot(np.array(obs_buffer)[0, obs_indices['goal_x']], np.array(obs_buffer)[0, obs_indices['goal_y']], 'ro', label='Goal')   # Goal
                curr_ax.set_xlim(ax_limits)
                curr_ax.set_ylim(ax_limits)
            
            axes = [ax[i, 5], ax_all[i, variant_idx]]
            for __ in range(len(sampled_trajectories_all[i])):          # Iterate over timesteps of sampled trajectories
                for ___ in range(min(args.batch_size, 4)):              # Iterate over batch
                    close_to_origin_threshold = 0.1 if 'pointmaze' in exp else 1
                    last_plot_index = np.where(~(np.linalg.norm(
                        sampled_trajectories_all[i][__][___, :, [obs_indices['x'], obs_indices['y']]].T - \
                        dataset.normalizer.normalizers['observations'].unnormalize(np.zeros(diffusion.observation_dim))[[obs_indices['x'], obs_indices['y']]],
                        axis=1) > close_to_origin_threshold))[0]
                    last_plot_index = args.horizon - 1 if len(last_plot_index) == 0 else last_plot_index[0]         # Ignore trajectory points at the end that are mapped to the origin (less than H steps to the goal)
                    for curr_ax in axes:
                        curr_ax.plot(sampled_trajectories_all[i][__][___, :last_plot_index, obs_indices['x']], sampled_trajectories_all[i][__][___, :last_plot_index, obs_indices['y']], 'b')
                        curr_ax.plot(sampled_trajectories_all[i][__][___, 0, obs_indices['x']], sampled_trajectories_all[i][__][___, 0, obs_indices['y']], 'go', label='Start')    # Current state
            ax[i, 5].plot(np.array(obs_buffer)[0, obs_indices['goal_x']], np.array(obs_buffer)[0, obs_indices['goal_y']], 'ro', label='Goal')   # Goal
            ax[i, 5].set_xlim(ax_limits)
            ax[i, 5].set_ylim(ax_limits)

            # Plot constraints
            axes = [ax[i, 4], ax[i, 5], ax_all[i, variant_idx]]
            for curr_ax in axes:
                if 'pointmaze' in exp:
                    curr_ax.add_patch(matplotlib.patches.Rectangle((-1.5, -0.5), 2, 1, color='k', alpha=0.2))
                else:
                    curr_ax.add_patch(matplotlib.patches.Rectangle((-6, -2), 8, 4, color='k', alpha=0.2))

                for constraint in constraint_points:
                    mat = np.zeros((3, 2))
                    mat[:2] = constraint[:2]
                    if 'pointmaze' in exp:
                        mat[2] = np.array([1.5, -1.5]) if constraint[2] == 'above' else np.array([1.5, 1.5])
                    elif 'antmaze' in exp:
                        mat[2] = np.array([6, -6]) if constraint[2] == 'above' else np.array([6, 6])
                    curr_ax.add_patch(matplotlib.patches.Polygon(mat, color='c', alpha=0.2))

        print(f'Success rate: {n_success / n_trials}')
        if n_success > 0:
            print(f'Average number of steps in successes: {n_steps / n_success}')
        print(f'Average computation time per step: {np.mean(avg_time)}')

        fig.savefig(f'./logs/umaze_plots/{exp}_{variant}.png')   

        ax_all[0, variant_idx].set_title(variant)
        env.close()

    fig_all.savefig(f'./logs/umaze_plots/{exp}.png')
    plt.show()