import time
import torch
from copy import copy
import minari
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import diffuser.utils as utils
from diffuser.sampling import Policy
from diffusers import DDPMScheduler, DDIMScheduler
from diffuser.sampling import Projector

# torch.set_default_device('cpu')

exps = [
    # 'pointmaze-umaze-dense-v2'
    'antmaze-umaze-v1',
    ]

projection_variants = [
    # 'none',
    # 'end_safe',     # Projected generative diffusion models
    # '0p1_safe',
    # '0p2_safe',
    # '0p5_safe',
    # 'full_safe',
    # 'end_all', 
    # 'end_all_cost',
    # 'end_all_cost_repeatlast',
    # '0p1_all',
    # '0p1_all_cost',
    '0p1_all_cost_repeatlast3',
    '0p1_all_cost_repeatlast5',
    # '0p2_all',
    '0p2_all_cost',
    '0p2_all_cost_repeatlast3',
    '0p2_all_cost_repeatlast5',
    # '0p5_all',
    '0p5_all_cost',
    '0p5_all_cost_repeatlast3',
    '0p5_all_cost_repeatlast5',
    # 'full_all',
    'full_all_cost',
    'full_all_cost_repeatlast3',
    'full_all_cost_repeatlast5',
    ]

for exp in exps:
    class Parser(utils.Parser):
        dataset: str = exp
        config: str = 'config.' + exp
    args = Parser().parse_args('plan')

    # Get model
    diffusion_experiment = utils.load_diffusion(
        args.loadbase, args.dataset, args.diffusion_loadpath,
        epoch=args.diffusion_epoch, seed=args.seed, device=args.device
    )

    diffusion_losses = diffusion_experiment.losses
    diffusion = diffusion_experiment.diffusion
    dataset = diffusion_experiment.dataset

    minari_dataset = minari.load_dataset(exp, download=True)
    env = minari_dataset.recover_environment(eval_env=True) if 'pointmaze' in exp else minari_dataset.recover_environment()    # Set render_mode='human' to visualize the environment

    # Create scheduler
    scheduler = DDIMScheduler(num_train_timesteps=diffusion.n_timesteps)   
    scheduler.set_timesteps(20)                         # Steps used for inference

    # Create projector
    if diffusion.__class__.__name__ == 'GaussianDiffusion':
        trajectory_dim = diffusion.transition_dim
        action_dim = diffusion.action_dim
    else:
        trajectory_dim = diffusion.observation_dim
        action_dim = 0

    if 'pointmaze' in exp:
        obs_indices = {'x': 0, 'y': 1, 'vx': 2, 'vy': 3, 'goal_x': 4, 'goal_y': 5}
        cost_dims = [obs_indices['x'], obs_indices['y'], obs_indices['vx'], obs_indices['vy']]
        if diffusion.__class__.__name__ == 'GaussianDiffusion': 
            # obs_indices = {k: v + diffusion.action_dim for k, v in obs_indices.items()}
            action_indices = {'ax': 0, 'ay': 1}
            cost_dims = [2, 3, 4, 5]      # Only position-velocity changes penalized  
    elif 'antmaze' in exp:
        obs_indices = {'x': 0, 'y': 1, 'z':2, 'qx': 3, 'qy': 4, 'qz': 5, 'qw': 6, 'hip1': 7, 'ankle1': 8, 'hip2': 9, 'ankle2': 10, 
                       'hip3': 11, 'ankle3': 12, 'hip4': 13, 'ankle4': 14, 'vx': 15, 'vy': 16, 'vz': 17, 'dhip1': 21, 'dankle1': 22,
                       'dhip2': 23, 'dankle2': 24, 'dhip3': 25, 'dankle3': 26, 'dhip4': 27, 'dankle4': 28, 'goal_x': 29, 'goal_y': 30, }
        cost_dims = [obs_indices['x'], obs_indices['y'], obs_indices['z'], obs_indices['vx'], obs_indices['vy'], obs_indices['vz']]
        # TODO: Account for action prediction case

    if 'pointmaze' in exp:
        safety_constraints = [
            [[0.25, -1.5], [1.5, -0.25], 'above'],
            [[1.5, 0.25], [0.25, 1.5], 'below'],
            ]
    else:
        safety_constraints = [
            # [[1, -6], [6, -1], 'above'],          # tight
            # [[6, 1], [1, 6], 'below'],            # |
            # [[1.5, -6], [6, -1.5], 'above'],      # |
            # [[6, 1.5], [1.5, 6], 'below'],        # v
            [[1.75, -6], [6, -1.75], 'above'],      # less tight
            [[6, 1.75], [1.75, 6], 'below'],
            # [[2, -6], [6, -2], 'above'],
            # [[6, 2], [2, 6], 'below'],
            ]
        
    constraint_list_safe = []
    for constraint in safety_constraints:
        m = (constraint[1][1] - constraint[0][1]) / (constraint[1][0] - constraint[0][0])
        d = constraint[0][1] - m * constraint[0][0]
        C_row = np.zeros(trajectory_dim)
        if constraint[2] == 'below':
            C_row[obs_indices['x'] + action_dim] = -m
            C_row[obs_indices['y'] + action_dim] = 1
        elif constraint[2] == 'above':
            C_row[obs_indices['x'] + action_dim] = m
            C_row[obs_indices['y'] + action_dim] = -1
            d *= -1
        constraint_list_safe.append(('ineq', (C_row, d)))
        
    # Enlarge safety constraints by tracking error
    tracking_error_bound = 0.25 if 'antmaze' in exp else 0.12
    safety_constraints_enlarged = []
    for constraint in safety_constraints:
        m = (constraint[1][1] - constraint[0][1]) / (constraint[1][0] - constraint[0][0])
        n = [-1, 1/m] / np.linalg.norm([-1, 1/m])
        safety_constraints_enlarged.append([constraint[0] + tracking_error_bound * n, constraint[1] + tracking_error_bound * n, constraint[2]])
    
    constraint_list_safe_enlarged = []
    for constraint in safety_constraints_enlarged:
        m = (constraint[1][1] - constraint[0][1]) / (constraint[1][0] - constraint[0][0])
        d = constraint[0][1] - m * constraint[0][0]
        C_row = np.zeros(trajectory_dim)
        if constraint[2] == 'below':
            C_row[obs_indices['x'] + action_dim] = -m
            C_row[obs_indices['y'] + action_dim] = 1
        elif constraint[2] == 'above':
            C_row[obs_indices['x'] + action_dim] = m
            C_row[obs_indices['y'] + action_dim] = -1
            d *= -1
        constraint_list_safe_enlarged.append(('ineq', (C_row, d)))

    constraint_list = copy(constraint_list_safe_enlarged)   # or copy(constraint_list_safe) to not enlarge the constraints
    if 'pointmaze' in exp:
        dynamic_constraints = [
            ('deriv', np.array([obs_indices['x'], obs_indices['vx']]) + action_dim),
            ('deriv', np.array([obs_indices['y'], obs_indices['vy']]) + action_dim),
        ]
    elif 'antmaze' in exp:
        dynamic_constraints = [
            ('deriv', np.array([obs_indices['x'], obs_indices['vx']]) + action_dim),
            ('deriv', np.array([obs_indices['y'], obs_indices['vy']]) + action_dim),
            ('deriv', np.array([obs_indices['z'], obs_indices['vz']]) + action_dim),
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
        constraint_list.append(constraint)

    if 'pointmaze' in exp:
        seeds = [7, 10, 11, 16, 24, 28, 31, 33, 39, 41, 43, 44, 45, 46, 48]     # Good seeds for pointmaze-umaze-dense-v2: [7, 10, 11, 16, 24, 28, 31, 33, 39, 41, 43, 44, 45, 46, 48]
        # seeds = [7, 10, 11, 16, 24]
    else:
        seeds = np.arange(5)                   
    n_trials = max(2, len(seeds))
    n_timesteps = 100 if 'pointmaze' in exp else 300

    fig_all, ax_all = plt.subplots(min(n_trials, 10), len(projection_variants), figsize=(20, 20))
    ax_limits = [-1.5, 1.5] if 'pointmaze' in exp else [-6, 6]

    for variant_idx, variant in enumerate(projection_variants):
        print(f'------------------------Running {exp} - {variant}----------------------------')

        if 'none' in variant:
            projector = None
        else:
            constraint_list_proj = constraint_list_safe_enlarged if 'safe' in variant else constraint_list
            if 'full' in variant:
                diffusion_timestep_threshold = 1
            elif '0p1' in variant:
                diffusion_timestep_threshold = 0.1
            elif '0p2' in variant:
                diffusion_timestep_threshold = 0.2
            elif '0p5' in variant:
                diffusion_timestep_threshold = 0.5
            else:
                diffusion_timestep_threshold = 0
            dt = 0.02 if 'pointmaze' in exp else 0.05
            projector = Projector(
                horizon=args.horizon, 
                transition_dim=trajectory_dim, 
                constraint_list=constraint_list_proj, 
                normalizer=dataset.normalizer, 
                diffusion_timestep_threshold=diffusion_timestep_threshold,
                dt=dt,
                cost_dims=cost_dims,
                device=args.device,
            )

        # Create policy
        return_costs = 'cost' in variant
        # repeat_last = 3 if 'repeatlast' in variant else 0
        if 'repeatlast1' in variant: repeat_last = 1
        elif 'repeatlast2' in variant: repeat_last = 2
        elif 'repeatlast3' in variant: repeat_last = 3
        elif 'repeatlast4' in variant: repeat_last = 4
        elif 'repeatlast5' in variant: repeat_last = 5
        else: repeat_last = 0
        policy = Policy(
            model=diffusion,
            scheduler=scheduler,
            normalizer=dataset.normalizer,
            preprocess_fns=args.preprocess_fns,
            test_ret=args.test_ret,
            projector=projector,
            # **sample_kwargs
            return_costs=return_costs,
            repeat_last=repeat_last,
        )    

        if 'pointmaze' in exp:
            env.env.env.env.point_env.frame_skip = 2
        if 'antmaze' in exp:
            env.env.env.env.ant_env.frame_skip = 5

        # Run policy
        fig, ax = plt.subplots(min(n_trials, 10), 6, figsize=(20, 20))
        fig.suptitle(f'{exp} - {variant}')

        action_update_every = 1
        save_samples_every = 10

        # Store a few sampled trajectories
        sampled_trajectories_all = []
        n_success = 0
        n_steps = 0
        n_violations = 0
        total_violations = 0
        avg_time = np.zeros(n_trials)
        for i in range(n_trials):
            torch.manual_seed(i)
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
                conditions = {0: obs}

                # Check if a safety constraint is violated
                for constraint in constraint_list_safe:
                    c, d = constraint[1]
                    if obs @ c[action_dim:] >= d + 1e-3:
                        n_violations += 1
                        total_violations += obs @ c[action_dim:] - d
                        break
                
                start = time.time()
                action, samples = policy(conditions, batch_size=args.batch_size, horizon=args.horizon, disable_projection=disable_projection)
                avg_time[i] += time.time() - start

                # Check whether one of the sampled trajectories violates a 
                disable_projection = True
                for constraint in constraint_list_safe_enlarged:
                    c, d = constraint[1]
                    if np.any(samples.observations @ c[action_dim:] >= d - 1e-2):   # (Close to) Violation of constraint
                        disable_projection = False
                        # print('Enabled projection at timestep', _)
                        break

                if _ % save_samples_every == 0:
                    sampled_trajectories.append(samples.observations[:, :, :])

                obs, rew, terminated, truncated, info = env.step(action)
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
                if info['success'] or terminated or truncated or _ == n_timesteps - 1:
                    n_steps += _
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

                for constraint in safety_constraints:
                    mat = np.zeros((3, 2))
                    mat[:2] = constraint[:2]
                    if 'pointmaze' in exp:
                        mat[2] = np.array([1.5, -1.5]) if constraint[2] == 'above' else np.array([1.5, 1.5])
                    elif 'antmaze' in exp:
                        mat[2] = np.array([6, -6]) if constraint[2] == 'above' else np.array([6, 6])
                    curr_ax.add_patch(matplotlib.patches.Polygon(mat, color='c', alpha=0.2))

        print(f'Success rate: {n_success / n_trials}')
        if n_success > 0:
            print(f'Avg number of steps: {n_steps / n_trials}')
            print(f'Avg number of constraint violations: {n_violations / n_trials}')
            print(f'Avg total violation: {total_violations / n_trials}')
        print(f'Average computation time per step: {np.mean(avg_time)}')

        fig.savefig(f'{args.savepath}/{variant}.png')   
        plt.close(fig)

        ax_all[0, variant_idx].set_title(variant)
        env.close()

    fig_all.savefig(f'{args.savepath}/all.png')
    plt.show()