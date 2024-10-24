import time
import yaml
import torch
from copy import copy
import minari
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import diffuser.utils as utils
from diffuser.sampling import Policy
from diffuser.sampling import Projector

# Load configuration
with open('config/projection_eval.yaml', 'r') as file:
    config = yaml.safe_load(file)

# General
exps = config['exps']
projection_variants = config['projection_variants']
n_trials = config['n_trials']

# Environmentfoal
dt_all = config['dt']
observation_indices = config['observation_indices']
seeds_all = config['seeds']

# Constraint projection
repeat_last = config['repeat_last']
diffusion_timestep_threshold = config['diffusion_timestep_threshold']
projection_cost = config['projection_cost']
constraint_types = config['constraint_types']
enlarge_constraints_all = config['enlarge_constraints']
halfspace_constraints = config['halfspace_constraints']
obstacle_constraints_all = config['obstacle_constraints']
bounds_all = config['bounds']

plot_how_many_combined = config['plot_how_many_combined']
plot_how_many_individual = config['plot_how_many_individual']
ax_limits_all = config['ax_limits']

for exp in exps:
    robot_name = 'pointmaze' if 'pointmaze' in exp else 'antmaze'
    dt = dt_all[robot_name]
    polytopic_constraints = halfspace_constraints[exp]
    enlarge_constraints = enlarge_constraints_all[robot_name]
    obstacle_constraints = obstacle_constraints_all[exp]
    bounds = bounds_all[exp]

    class Parser(utils.Parser):
        dataset: str = exp
        config: str = 'config.' + exp
    args = Parser().parse_args('plan')

    # Get model
    diffusion_experiment = utils.load_diffusion(
        args.loadbase, args.dataset, args.diffusion_loadpath,
        epoch=args.diffusion_epoch, seed=args.seed, device=args.device
    )
    diffusion = diffusion_experiment.diffusion
    dataset = diffusion_experiment.dataset

    minari_dataset = minari.load_dataset(exp, download=True)
    env = minari_dataset.recover_environment(eval_env=True) if 'pointmaze' in exp else minari_dataset.recover_environment()    # Set render_mode='human' to visualize the environment
    if robot_name == 'pointmaze':
        env.env.env.env.point_env.frame_skip = 2
    if robot_name == 'antmaze':
        env.env.env.env.ant_env.frame_skip = 5

    # Create projector
    if diffusion.__class__.__name__ == 'GaussianDiffusion':
        # trajectory_dim = diffusion.transition_dim
        trajectory_dim = diffusion.transition_dim - diffusion.goal_dim
        action_dim = diffusion.action_dim
        diffuser_variant = 'states_actions'
    else:
        # trajectory_dim = diffusion.observation_dim
        trajectory_dim = diffusion.observation_dim - diffusion.goal_dim
        action_dim = 0
        diffuser_variant = 'states'

    obs_indices = observation_indices[robot_name]
    if robot_name == 'pointmaze':
        cost_dims = [obs_indices['x'], obs_indices['y'], obs_indices['vx'], obs_indices['vy']] if projection_cost == 'pos_vel' else [obs_indices['x'], obs_indices['y']]      
    elif robot_name == 'antmaze':
        cost_dims = [obs_indices['x'], obs_indices['y'], obs_indices['z'], obs_indices['vx'], obs_indices['vy'], obs_indices['vz']]

    if diffusion.__class__.__name__ == 'GaussianDiffusion': 
        for dim in cost_dims:
            cost_dims[dim] += action_dim

    # -------------------- Load constraints ------------------
    # Halfspace constraints
    constraint_list = []
    constraint_list_polytopic_not_enlarged = []
    if 'halfspace' in constraint_types:
        for constraint in polytopic_constraints:
            C_row, d = utils.formulate_halfspace_constraints(constraint, enlarge_constraints, trajectory_dim, action_dim, obs_indices)
            constraint_list.append(('ineq', (C_row, d)))
            C_row, d = utils.formulate_halfspace_constraints(constraint, 0, trajectory_dim, action_dim, obs_indices)
            constraint_list_polytopic_not_enlarged.append(('ineq', (C_row, d)))

    # Bounds
    lower_bound = -np.inf * np.ones(trajectory_dim)
    upper_bound = np.inf * np.ones(trajectory_dim)
    if 'bounds' in constraint_types:
        for bound in bounds:
            for dim_idx, dim in enumerate(bound['dimensions']):
                if bound['type'] == 'lower':
                    lower_bound[obs_indices[dim]] = bound['values'][dim_idx]
                elif bound['type'] == 'upper':
                    upper_bound[obs_indices[dim]] = bound['values'][dim_idx]
        constraint_list.append(('lb', lower_bound))
        constraint_list.append(('ub', upper_bound))

    # Obstacle constraints
    if 'obstacles' in constraint_types:
        for constraint in obstacle_constraints:
            constraint_list.append([
                constraint['type'], 
                [obs_indices[constraint['dimensions'][0]] + action_dim, obs_indices[constraint['dimensions'][1]] + action_dim], 
                constraint['center'], 
                constraint['radius'] + enlarge_constraints
            ])
    
    # Dynamics constraints
    constraint_list_without_dynamics = copy(constraint_list)
    if 'dynamics' in constraint_types:
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
            ]

        for constraint in dynamic_constraints:
            constraint_list.append(constraint)

    # -------------------- Run experiments ------------------
    seeds = seeds_all[exp] if 'pointmaze-umaze' in exp else np.arange(100)
    seeds = seeds[:n_trials]

    n_timesteps = args.max_episode_length
               
    fig_all, ax_all = plt.subplots(min(n_trials, plot_how_many_combined), len(projection_variants), figsize=(10 * len(projection_variants), 10 * min(n_trials, plot_how_many_combined)))
    ax_limits = ax_limits_all[exp]

    for variant_idx, variant in enumerate(projection_variants):
        print(f'------------------------Running {exp} - {variant}----------------------------')

        threshold = diffusion_timestep_threshold if not 'post_processing' in variant else 0
        constraints = constraint_list if not 'no_dynamics' in variant else constraint_list_without_dynamics
        lb, ub = lower_bound, upper_bound if not 'no_bounds' in variant else None
        # Create projector
        projector = Projector(
            horizon=args.horizon, 
            transition_dim=trajectory_dim, 
            goal_dim=diffusion.goal_dim,
            constraint_list=constraints, 
            normalizer=dataset.normalizer, 
            diffusion_timestep_threshold=threshold,
            variant=diffuser_variant,
            dt=dt,
            cost_dims=cost_dims,
            device=args.device,
            solver='scipy',
        )
        projector = None if variant == 'diffuser' else projector

        trajectory_selection = 'random'
        if 'consistency' in variant: trajectory_selection = 'temporal_consistency'
        if 'costs' in variant: trajectory_selection = 'minimum_projection_cost'
        repeat_last = 2 if 'repeat' in variant else 0

        # Create policy
        policy = Policy(
            model=diffusion,
            normalizer=dataset.normalizer,
            preprocess_fns=args.preprocess_fns,
            test_ret=args.test_ret,
            projector=projector,
            trajectory_selection=trajectory_selection,
            # **sample_kwargs
            repeat_last=repeat_last,
        )    

        # Run policy
        fig, ax = plt.subplots(min(n_trials, plot_how_many_individual), 6, figsize=(30, 5 * min(n_trials, plot_how_many_individual)))
        fig.suptitle(f'{exp} - {variant}')

        action_update_every = 1
        save_samples_every = args.horizon // 2

        # Store a few sampled trajectories
        sampled_trajectories_all = []        
        n_success = np.zeros(n_trials)
        n_steps = np.zeros(n_trials)
        n_violations = np.zeros(n_trials)
        total_violations = np.zeros(n_trials)
        avg_time = np.zeros(n_trials)
        collision_free_completed = np.ones(n_trials)
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
            # disable_projection = True
            disable_projection = False
            for _ in range(n_timesteps):
                # print(f'Trial {i}, timestep {_}')
                conditions = {0: obs}

                # Check if a safety constraint is violated
                violated_this_timestep = 0
                if 'halfspace' in constraint_types:
                    for constraint in constraint_list_polytopic_not_enlarged:
                        if constraint[0] == 'ineq':
                            c, d = constraint[1]
                            if obs[:-diffusion.goal_dim] @ c[action_dim:] >= d:
                                violated_this_timestep = 1
                                total_violations[i] += obs[:-diffusion.goal_dim] @ c[action_dim:] - d
                                collision_free_completed[i] = 0
                                break
                if 'obstacles' in constraint_types:
                    for constraint in obstacle_constraints:
                        if np.linalg.norm(obs[[obs_indices['x'], obs_indices['y']]] - constraint['center']) < constraint['radius']:
                            violated_this_timestep = 1
                            total_violations[i] += constraint['radius'] - np.linalg.norm(obs[[obs_indices['x'], obs_indices['y']]] - constraint['center'])
                            collision_free_completed[i] = 0
                            break
                n_violations[i] += violated_this_timestep
                
                start = time.time()
                action, samples = policy(conditions, batch_size=args.batch_size, horizon=args.horizon, disable_projection=disable_projection)
                avg_time[i] += time.time() - start

                # Check whether one of the sampled trajectories violates a constraint
                # disable_projection = True
                disable_projection = False
                for constraint in constraint_list:
                    if constraint[0] == 'lb' and np.any(samples.observations[:, 1:, :-diffusion.goal_dim] < constraint[1][action_dim:] - 1e-3) and variant != 'diffuser':
                        print('Predicted trajectory violates lower bound')
                    if constraint[0] == 'ub' and np.any(samples.observations[:, 1:, :-diffusion.goal_dim] > constraint[1][action_dim:] + 1e-3) and variant != 'diffuser':
                        print('Predicted trajectory violates upper bound')
                    
                    if constraint[0] == 'ineq':
                        c, d = constraint[1]
                        if np.any(samples.observations[:,:,:-diffusion.goal_dim] @ c[action_dim:] >= d):   # (Close to) Violation of constraint
                            disable_projection = False
                            break
                if 'obstacles' in constraint_types:
                    for constraint in obstacle_constraints:
                        if np.any(np.linalg.norm(samples.observations[:, :, [obs_indices['x'], obs_indices['y']]] - constraint['center'], axis=-1) < constraint['radius'] + enlarge_constraints):
                            disable_projection = False
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
                        collision_free_completed[i] = 0
                    if dist_to_goal <= 1:
                        info['success'] = True

                obs_buffer.append(obs)
                action_buffer.append(action)
                if info['success']:
                    n_success[i] = 1
                if info['success'] or terminated or truncated or _ == n_timesteps - 1:
                    n_steps[i] = _
                    avg_time[i] /= _
                    break

            sampled_trajectories_all.append(sampled_trajectories)
            if i >= plot_how_many_individual:     # Plot only the first n trials
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
                if exp == 'pointmaze-umaze-dense-v2':
                    curr_ax.add_patch(matplotlib.patches.Rectangle((-1.5, -0.5), 2, 1, color='k', alpha=0.2))
                if exp == 'pointmaze-medium-dense-v2':
                    bottom_left_corners = [[-1, 2], [0, 2], [-1, 1], [-3, 0], [1, 0], [2, 0], [-1, -1], [-2, -2], [1, -2], [0, -3]]
                    for corner in bottom_left_corners:
                        curr_ax.add_patch(matplotlib.patches.Rectangle((corner[0], corner[1]), 1, 1, color='k', alpha=0.2))
                elif exp == 'antmaze-umaze-v1':
                    curr_ax.add_patch(matplotlib.patches.Rectangle((-6, -2), 8, 4, color='k', alpha=0.2))

                if 'halfspace' in constraint_types:
                    for constraint in polytopic_constraints:
                        mat = np.vstack((constraint[:2], np.zeros(2)))
                        if 'pointmaze' in exp:
                            mat[2] = np.array([1.5, -1.5]) if constraint[2] == 'above' else np.array([1.5, 1.5])
                        elif 'antmaze' in exp:
                            mat[2] = np.array([6, -6]) if constraint[2] == 'above' else np.array([6, 6])
                        curr_ax.add_patch(matplotlib.patches.Polygon(mat, color='m', alpha=0.2))

                if 'obstacles' in constraint_types:
                    for constraint in obstacle_constraints:
                        curr_ax.add_patch(matplotlib.patches.Circle(constraint['center'], constraint['radius'], color='m', alpha=0.2))

        print(f'Success rate: {np.mean(n_success)}')
        print(f'Avg number of steps: {np.mean(n_steps):.2f} +- {np.std(n_steps):.2f}')
        print(f'Avg number of constraint violations: {np.mean(n_violations):.2f} +- {np.std(n_violations):.2f}')
        print(f'Avg total violation: {np.mean(total_violations):.3f} +- {np.std(total_violations):.3f}')
        print(f'Average computation time per step: {np.mean(avg_time):.3f}')
        print(f'Collision free completed: {np.mean(collision_free_completed)}')
        if config['write_to_file']:
            with open(f'{args.savepath}/results.txt', 'a') as file:
                file.write(f'{exp} - {variant}\n')
                file.write(f'SR: {np.mean(n_success)}, Avg steps: {np.mean(n_steps):.2f}, Avg violations: {np.mean(n_violations):.2f}, Avg total violation: {np.mean(total_violations):.3f}, Avg time: {np.mean(avg_time):.3f}, Collision free completed: {collision_free_completed.sum()} / {n_trials}\n')

        fig.savefig(f'{args.savepath}/{variant}.png')   
        plt.close(fig)

        ax_all[0, variant_idx].set_title(variant)
        env.close()

    fig_all.savefig(f'{args.savepath}/all.png')
    plt.show()