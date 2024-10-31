import time
import yaml
import os
import torch
from copy import copy
import minari
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import diffuser.utils as utils
from diffuser.sampling import Policy, Projector
from d3il.environments.d3il.envs.gym_avoiding_env.gym_avoiding.envs.avoiding import ObstacleAvoidanceEnv

# Load configuration
with open('config/projection_eval.yaml', 'r') as file:
    config = yaml.safe_load(file)

# General
exps = config['exps']
seeds = config['seeds']
projection_variants = config['projection_variants']
n_trials = config['n_trials']
plot_how_many = config['plot_how_many']

# Constraint projection
repeat_last = config['repeat_last']
diffusion_timestep_threshold = config['diffusion_timestep_threshold']
constraint_types = config['constraint_types']

for exp in exps:
    results = {}

    robot_name = exp.split('-')[0]
    polytopic_constraints = config['halfspace_constraints'][exp]
    obstacle_constraints = config['obstacle_constraints'][exp]
    bounds = config['bounds'][exp]
    ax_limits = config['ax_limits'][exp]
    enlarge_constraints = config['enlarge_constraints'][robot_name]
    dt = config['dt'][robot_name]

    class Parser(utils.Parser):
        dataset: str = exp
        config: str = 'config.' + exp

    for seed in seeds:
        args = Parser().parse_args(experiment='plan', seed=seed)

        # Get model
        diffusion_experiment = utils.load_diffusion(args.loadbase, args.dataset, args.diffusion_loadpath, str(args.seed), epoch=args.diffusion_epoch, device=args.device)
        diffusion = diffusion_experiment.diffusion
        dataset = diffusion_experiment.dataset

        if 'pointmaze' in exp or 'antmaze' in exp:
            minari_dataset = minari.load_dataset(exp, download=True)
            env = minari_dataset.recover_environment(eval_env=True) if 'pointmaze' in exp else minari_dataset.recover_environment()    # Set render_mode='human' to visualize the environment
        elif 'avoiding' in exp:
            env = ObstacleAvoidanceEnv()
            env.start()

        if robot_name == 'pointmaze': env.env.env.env.point_env.frame_skip = 2
        if robot_name == 'antmaze': env.env.env.env.ant_env.frame_skip = 5

        obs_indices = config['observation_indices'][robot_name]
        act_indices = config['action_indices'][robot_name]

        # Create projector
        if diffusion.__class__.__name__ == 'GaussianDiffusion':
            trajectory_dim = diffusion.transition_dim - diffusion.goal_dim
            action_dim = diffusion.action_dim
            diffuser_variant = 'states_actions'
            obs_indices_updated = {key: val + action_dim for key, val in obs_indices.items()}
            act_obs_indices = {**act_indices, **obs_indices_updated}
        else:
            trajectory_dim = diffusion.observation_dim - diffusion.goal_dim
            action_dim = 0
            diffuser_variant = 'states'
            act_obs_indices = obs_indices

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
        constraint_list_without_prior = copy(constraint_list)
        dynamics_constraints = []
        if 'dynamics' in constraint_types: dynamics_constraints = utils.formulate_dynamics_constraints(exp, act_obs_indices, action_dim)

        for constraint in dynamics_constraints:
            constraint_list.append(constraint)

        # -------------------- Run experiments ------------------
        env_seeds = config['env_seeds'][exp] if 'pointmaze-umaze' in exp else np.arange(100)       
        fig_all, ax_all = plt.subplots(min(n_trials, plot_how_many), len(projection_variants), figsize=(10 * len(projection_variants), 10 * min(n_trials, plot_how_many)))

        for variant_idx, variant in enumerate(projection_variants):
            print(f'------------------------Running {exp} - {variant} ({seed})----------------------------')

            threshold = diffusion_timestep_threshold if not 'post_processing' in variant else 0

            constraints = constraint_list if not 'no_prior' in variant else constraint_list_without_prior
            # Create projector
            projector = Projector(horizon=args.horizon, transition_dim=trajectory_dim, action_dim=action_dim, goal_dim=diffusion.goal_dim, constraint_list=constraints, normalizer=dataset.normalizer, 
                                    diffusion_timestep_threshold=threshold, variant=diffuser_variant, dt=dt, cost_dims=None, device=args.device, solver='scipy')        # takes 0.02s
            projector = None if variant == 'diffuser' else projector

            trajectory_selection = 'random'
            if 'consistency' in variant: trajectory_selection = 'temporal_consistency'
            if 'costs' in variant: trajectory_selection = 'minimum_projection_cost'
            repeat_last = 2 if 'repeat' in variant else 0
            project_x_recon = True if not 'project_x_t' in variant else False

            # Create policy
            policy = Policy(
                model=diffusion, normalizer=dataset.normalizer, preprocess_fns=args.preprocess_fns, test_ret=args.test_ret, projector=projector, trajectory_selection=trajectory_selection,
                # **sample_kwargs
                repeat_last=repeat_last, project_x_recon=project_x_recon,
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
            pos_tracking_errors = np.zeros((n_trials, args.max_episode_length - 1))
            for i in range(n_trials):
                torch.manual_seed(i)
                env_seed = env_seeds[i] if ('pointmaze-umaze' in exp) else i
                
                # Reset environment
                if 'avoiding' in exp:
                    obs = env.reset()
                    action = env.robot_state()[:2]
                    fixed_z = env.robot_state()[2:]
                else:
                    obs, _ = env.reset(seed=env_seed)
                
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
                        # if np.sum(np.maximum(0, act_obs - upper_bound)) + np.sum(np.maximum(0, lower_bound - act_obs)) > 0:
                        #     print('Bounds violated')
                        # violated_this_timestep = 1

                    n_violations[i] += violated_this_timestep
                    
                    # Calculate action
                    start = time.time()
                    action, samples = policy(conditions={0: obs}, batch_size=args.batch_size, horizon=args.horizon, disable_projection=disable_projection)
                    avg_time[i] += time.time() - start

                    # Step environment
                    if 'avoiding' in exp:
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

                    # Get tracking error
                    if _ >= 1:
                        pos_tracking_errors[i, _-1] = np.linalg.norm(obs[obs_indices['x']:obs_indices['y']+1] - desired_next_pos)
                    desired_next_pos = samples.observations[0, 1, [obs_indices['x'], obs_indices['y']]]

                    if _ % save_samples_every == 0:
                        sampled_trajectories.append(samples.observations[:, :, :])

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
                os.makedirs(f'{args.savepath}/results', exist_ok=True)
                # Save to .npz
                np.savez(f'{args.savepath}/results/{variant}.npz', 
                        n_success=n_success, 
                        n_steps=n_steps, 
                        n_violations=n_violations, 
                        total_violations=total_violations, 
                        avg_time=avg_time, 
                        collision_free_completed=collision_free_completed, 
                        args=args)

            fig.savefig(f'{args.savepath}/{variant}.png')   
            plt.close(fig)

            ax_all[0, variant_idx].set_title(variant)
            env.close()

        fig_all.savefig(f'{args.savepath}/all.png')
        plt.show()