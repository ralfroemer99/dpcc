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
from d3il.environments.d3il.envs.gym_avoiding_env.gym_avoiding.envs.avoiding import ObstacleAvoidanceEnv

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
action_indices = config['action_indices']
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

plot_how_many = config['plot_how_many']
ax_limits_all = config['ax_limits']

for exp in exps:
    if 'pointmaze' in exp: robot_name = 'pointmaze'
    if 'antmaze' in exp: robot_name = 'antmaze'
    if 'avoiding' in exp: robot_name = 'avoiding'
    dt = dt_all[robot_name]
    polytopic_constraints = halfspace_constraints[exp]
    enlarge_constraints = enlarge_constraints_all[robot_name] if robot_name in enlarge_constraints_all else 0
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

    if 'pointmaze' in exp or 'antmaze' in exp:
        minari_dataset = minari.load_dataset(exp, download=True)
        env = minari_dataset.recover_environment(eval_env=True) if 'pointmaze' in exp else minari_dataset.recover_environment()    # Set render_mode='human' to visualize the environment
    elif 'd3il-avoiding' in exp:
        env = ObstacleAvoidanceEnv()
        env.start()

    if robot_name == 'pointmaze':
        env.env.env.env.point_env.frame_skip = 2
    if robot_name == 'antmaze':
        env.env.env.env.ant_env.frame_skip = 5

    obs_indices = observation_indices[robot_name]
    act_indices = action_indices[robot_name]

    # Create projector
    if diffusion.__class__.__name__ == 'GaussianDiffusion':
        # trajectory_dim = diffusion.transition_dim
        trajectory_dim = diffusion.transition_dim - diffusion.goal_dim
        action_dim = diffusion.action_dim
        diffuser_variant = 'states_actions'
        obs_indices_updated = {key: val + action_dim for key, val in obs_indices.items()}
        act_obs_indices = {**act_indices, **obs_indices_updated}
    else:
        # trajectory_dim = diffusion.observation_dim
        trajectory_dim = diffusion.observation_dim - diffusion.goal_dim
        action_dim = 0
        diffuser_variant = 'states'
        act_obs_indices = obs_indices

    # if robot_name == 'pointmaze':
    #     # cost_dims = [obs_indices['x'], obs_indices['y'], obs_indices['vx'], obs_indices['vy']] if projection_cost == 'pos_vel' else [obs_indices['x'], obs_indices['y']]   
    #     cost_dims = [act_obs_indices['x'], act_obs_indices['y'], act_obs_indices['vx'], act_obs_indices['vy']] if projection_cost == 'pos_vel' else [act_obs_indices['x'], act_obs_indices['y']]
    # elif robot_name == 'antmaze':
    #     # cost_dims = [obs_indices['x'], obs_indices['y'], obs_indices['z'], obs_indices['vx'], obs_indices['vy'], obs_indices['vz']]
    #     cost_dims = [act_obs_indices['x'], act_obs_indices['y'], act_obs_indices['z'], act_obs_indices['vx'], act_obs_indices['vy'], act_obs_indices['vz']]
    # elif robot_name == 'avoiding':
    #     # cost_dims = [obs_indices['x_des'], obs_indices['y_des'], obs_indices['x'], obs_indices['y']]
    #     cost_dims = [act_obs_indices['x_des'], act_obs_indices['y_des'], act_obs_indices['x'], act_obs_indices['y']]

    # if diffusion.__class__.__name__ == 'GaussianDiffusion': 
    #     for dim in cost_dims:
    #         cost_dims[dim] += action_dim

    # -------------------- Load constraints ------------------
    # Halfspace constraints
    constraint_list = []
    constraint_list_polytopic_not_enlarged = []
    if 'halfspace' in constraint_types:
        for constraint in polytopic_constraints:
            constraint_list.append(('ineq', utils.formulate_halfspace_constraints(constraint, enlarge_constraints, trajectory_dim, act_obs_indices)))
            constraint_list_polytopic_not_enlarged.append(('ineq', utils.formulate_halfspace_constraints(constraint, 0, trajectory_dim, act_obs_indices)))

    # Bounds
    if 'bounds' in constraint_types:
        lower_bound, upper_bound = utils.formulate_bounds_constraints(constraint_types, bounds, trajectory_dim, act_obs_indices)
        constraint_list.extend([['lb', lower_bound], ['ub', upper_bound]])

    # Obstacle constraints
    if 'obstacles' in constraint_types:
        for constr in obstacle_constraints:
            constraint_list.append([constr['type'], [act_obs_indices[constr['dimensions'][0]], act_obs_indices[constr['dimensions'][1]]], constr['center'], constr['radius'] + enlarge_constraints])

    # Dynamics constraints
    constraint_list_without_dynamics = copy(constraint_list)
    dynamics_constraints = []
    if 'dynamics' in constraint_types: dynamics_constraints = utils.formulate_dynamics_constraints(exp, act_obs_indices, action_dim)

    for constraint in dynamics_constraints:
        constraint_list.append(constraint)

    # -------------------- Run experiments ------------------
    seeds = seeds_all[exp] if 'pointmaze-umaze' in exp else np.arange(100)
    seeds = seeds[:n_trials]
               
    fig_all, ax_all = plt.subplots(min(n_trials, plot_how_many), len(projection_variants), figsize=(10 * len(projection_variants), 10 * min(n_trials, plot_how_many)))
    ax_limits = ax_limits_all[exp]

    for variant_idx, variant in enumerate(projection_variants):
        print(f'------------------------Running {exp} - {variant}----------------------------')

        threshold = diffusion_timestep_threshold if not 'post_processing' in variant else 0
        constraints = constraint_list if not 'no_dynamics' in variant else constraint_list_without_dynamics
        # lb, ub = lower_bound, upper_bound if not 'no_bounds' in variant else None
        # Create projector
        projector = Projector(horizon=args.horizon, transition_dim=trajectory_dim, action_dim=action_dim, goal_dim=diffusion.goal_dim, constraint_list=constraints, normalizer=dataset.normalizer, 
                                diffusion_timestep_threshold=threshold, variant=diffuser_variant, dt=dt, cost_dims=None, device=args.device, solver='scipy')        # takes 0.02s
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
        fig, ax = plt.subplots(min(n_trials, plot_how_many), 6, figsize=(30, 5 * min(n_trials, plot_how_many)))
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
            
            # Reset environment
            if 'd3il-avoiding' in exp:
                obs = env.reset()
                action = env.robot_state()[:2]
                fixed_z = env.robot_state()[2:]
            else:
                obs, _ = env.reset(seed=seed)
            
            if 'pointmaze' in exp:
                obs = np.concatenate((obs['observation'], obs['desired_goal']))
            elif 'antmaze' in exp:
                obs = np.concatenate((obs['achieved_goal'], obs['observation'], obs['desired_goal']))
            elif 'avoiding' in exp:
                obs = np.concatenate((action[:2], obs))           
                
            obs_buffer = []
            action_buffer = []

            sampled_trajectories = []
            # disable_projection = True
            disable_projection = False
            for _ in range(args.max_episode_length):
                # print(f'Trial {i}, timestep {_}')

                # Check if a safety constraint is violated
                violated_this_timestep = 0
                if 'halfspace' in constraint_types:
                    for constraint in constraint_list_polytopic_not_enlarged:
                        if constraint[0] == 'ineq':
                            c, d = constraint[1]
                            obs_to_check = obs[:-diffusion.goal_dim] if diffusion.goal_dim > 0 else obs
                            if obs_to_check @ c[action_dim:] >= d:
                                violated_this_timestep = 1
                                total_violations[i] += obs_to_check @ c[action_dim:] - d
                                collision_free_completed[i] = 0

                if 'obstacles' in constraint_types:
                    for constraint in obstacle_constraints:
                        if np.linalg.norm(obs[[obs_indices['x'], obs_indices['y']]] - constraint['center']) < constraint['radius']:
                            violated_this_timestep = 1
                            total_violations[i] += constraint['radius'] - np.linalg.norm(obs[[obs_indices['x'], obs_indices['y']]] - constraint['center'])
                            collision_free_completed[i] = 0
                
                if _ > 0 and 'bounds' in constraint_types:
                    # if np.any(obs[:-diffusion.goal_dim] < lb - 1e-3) or np.any(obs[:-diffusion.goal_dim] > ub + 1e-3):
                    act_obs = np.concatenate((action, obs)) if action_dim > 0 else obs
                    total_violations[i] += np.sum(np.maximum(0, act_obs - upper_bound)) + np.sum(np.maximum(0, lower_bound - act_obs))
                    if np.sum(np.maximum(0, act_obs - upper_bound)) + np.sum(np.maximum(0, lower_bound - act_obs)) > 0:
                        print('Bounds violated')
                    violated_this_timestep = 1

                n_violations[i] += violated_this_timestep
                
                # Calculate action
                start = time.time()
                action, samples = policy(conditions={0: obs}, batch_size=args.batch_size, horizon=args.horizon, disable_projection=disable_projection)
                avg_time[i] += time.time() - start

                # Step environment
                if 'd3il-avoiding' in exp:
                    next_pos_des = action + obs[:2] 
                    obs, rew, terminated, info = env.step(np.concatenate((next_pos_des, fixed_z, [0, 1, 0, 0]), axis=0))
                    success = info[1]
                else:
                    obs, rew, terminated, truncated, info = env.step(action)
                    success = info['success']

                if 'pointmaze' in exp:
                    obs = np.concatenate((obs['observation'], obs['desired_goal']))
                elif 'antmaze' in exp:
                    obs = np.concatenate((obs['achieved_goal'], obs['observation'], obs['desired_goal']))
                elif 'avoiding' in exp:
                    obs = np.concatenate((next_pos_des[:2], obs))

                # Check whether one of the sampled trajectories violates a constraint
                # disable_projection = True
                disable_projection = False
                # for constraint in constraint_list:
                #     obs_to_check = samples.observations[:, 1:, :-diffusion.goal_dim] if diffusion.goal_dim > 0 else samples.observations[:, 1:]
                #     if constraint[0] == 'lb' and np.any(obs_to_check < constraint[1][action_dim:] - 1e-3) and variant != 'diffuser':
                #         print('Predicted trajectory violates lower bound')
                #     if constraint[0] == 'ub' and np.any(obs_to_check > constraint[1][action_dim:] + 1e-3) and variant != 'diffuser':
                #         print('Predicted trajectory violates upper bound')
                    
                #     if constraint[0] == 'ineq':
                #         c, d = constraint[1]
                #         obs_to_check = samples.observations[:, :, :-diffusion.goal_dim] if diffusion.goal_dim > 0 else samples.observations
                #         if np.any(obs_to_check @ c[action_dim:] >= d):   # (Close to) Violation of constraint
                #             disable_projection = False
                #             break
                # if 'obstacles' in constraint_types:
                #     for constraint in obstacle_constraints:
                #         if np.any(np.linalg.norm(samples.observations[:, :, [obs_indices['x'], obs_indices['y']]] - constraint['center'], axis=-1) < constraint['radius'] + enlarge_constraints):
                #             disable_projection = False
                #             break

                if _ % save_samples_every == 0:
                    sampled_trajectories.append(samples.observations[:, :, :])

                # For ant robot, check if it has flipped over or reached the goal (not provided by the environment)
                if 'antmaze' in exp:
                    quat = [obs[obs_indices['qx']], obs[obs_indices['qy']], obs[obs_indices['qz']], obs[obs_indices['qw']]]
                    if obs[obs_indices['z']] < 0.3:    # Ant is likely flipped over
                        terminated = True
                        collision_free_completed[i] = 0
                    if np.linalg.norm(obs['achieved_goal'] - obs['desired_goal']) <= 1:
                        success = True

                obs_buffer.append(obs)
                action_buffer.append(action)
                if success: n_success[i] = 1
                if terminated and (not success): collision_free_completed[i] = 0
                # if success or terminated or truncated or _ == n_timesteps - 1:
                if success or terminated or _ == args.max_episode_length - 1:
                    n_steps[i] = _
                    avg_time[i] /= _
                    break

            sampled_trajectories_all.append(sampled_trajectories)
            if i >= plot_how_many:     # Plot only the first n trials
                continue
            plot_states = ['x', 'y', 'vx', 'vy'] if 'maze' in exp else ['x', 'y']

            for j in range(len(plot_states)):
                ax[i, j].plot(np.array(obs_buffer)[:, obs_indices[plot_states[j]]])
                ax[i, j].set_title(plot_states[j])
            
            axes = [ax[i, 4], ax_all[i, variant_idx]]
            for curr_ax in axes:
                curr_ax.plot(np.array(obs_buffer)[:, obs_indices['x']], np.array(obs_buffer)[:, obs_indices['y']], 'k')
                curr_ax.plot(np.array(obs_buffer)[0, obs_indices['x']], np.array(obs_buffer)[0, obs_indices['y']], 'go', label='Start')            # Start
                if 'maze' in exp: curr_ax.plot(np.array(obs_buffer)[0, obs_indices['goal_x']], np.array(obs_buffer)[0, obs_indices['goal_y']], 'ro', label='Goal')   # Goal
                curr_ax.set_xlim(ax_limits[0])
                curr_ax.set_ylim(ax_limits[1])
            
            axes = [ax[i, 5], ax_all[i, variant_idx]]
            for __ in range(len(sampled_trajectories_all[i])):          # Iterate over timesteps of sampled trajectories
                for ___ in range(min(args.batch_size, 4)):              # Iterate over batch
                    for curr_ax in axes:
                        curr_ax.plot(sampled_trajectories_all[i][__][___, :args.horizon, obs_indices['x']], sampled_trajectories_all[i][__][___, :args.horizon, obs_indices['y']], 'b')
                        curr_ax.plot(sampled_trajectories_all[i][__][___, 0, obs_indices['x']], sampled_trajectories_all[i][__][___, 0, obs_indices['y']], 'go', label='Start')    # Current state
            if 'maze' in exp: ax[i, 5].plot(np.array(obs_buffer)[0, obs_indices['goal_x']], np.array(obs_buffer)[0, obs_indices['goal_y']], 'ro', label='Goal')   # Goal
            ax[i, 5].set_xlim(ax_limits[0])
            ax[i, 5].set_ylim(ax_limits[1])

            # Plot constraints
            axes = [ax[i, 4], ax[i, 5], ax_all[i, variant_idx]]
            for curr_ax in axes: 
                utils.plot_environment_constraints(exp, curr_ax)

                if 'halfspace' in constraint_types: utils.plot_halfspace_constraints(exp, polytopic_constraints, curr_ax, ax_limits)

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