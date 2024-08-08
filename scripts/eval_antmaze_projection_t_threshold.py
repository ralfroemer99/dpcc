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

exp = 'antmaze-umaze-v1'

dynamics_constraints_variants = [
    'pos_only',
    # 'pos_legs',
    ]

diffusion_timestep_thresholds = [-1, 0, 0.05, 0.1, 0.15, 0.2, 0.25]

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

# Create scheduler
scheduler = DDIMScheduler(num_train_timesteps=diffusion.n_timesteps)   
scheduler.set_timesteps(20)                         # Steps used for inference

# Create projector
if diffusion.__class__.__name__ == 'GaussianDiffusion':
    trajectory_dim = diffusion.transition_dim
else:
    trajectory_dim = diffusion.observation_dim

obs_indices = {'x': 0, 'y': 1, 'z':2, 'qx': 3, 'qy': 4, 'qz': 5, 'qw': 6, 'hip1': 7, 'ankle1': 8, 'hip2': 9, 'ankle2': 10, 
                'hip3': 11, 'ankle3': 12, 'hip4': 13, 'ankle4': 14, 'vx': 15, 'vy': 16, 'vz': 17, 'dhip1': 21, 'dankle1': 22,
                'dhip2': 23, 'dankle2': 24, 'dhip3': 25, 'dankle3': 26, 'dhip4': 27, 'dankle4': 28, 'goal_x': 29, 'goal_y': 30, }
cost_dims = [obs_indices['x'], obs_indices['y'], obs_indices['z'], obs_indices['vx'], obs_indices['vy'], obs_indices['vz']]

safety_constraints = [
    [[1, -6], [6, -1], 'above'], [[6, 1], [1, 6], 'below'],
    # [[1.5, -6], [6, -1.5], 'above'], [[6, 1.5], [1.5, 6], 'below'],     # Less tight
    ]

for dynamic_constraint_variant in dynamics_constraints_variants:
    constraint_list = []
    for constraint in safety_constraints:
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
        constraint_list.append(('ineq', (C_row, d)))
    constraint_list_safe = copy(constraint_list)

    if dynamic_constraint_variant == 'pos_only':
        dynamic_constraints = [
            ('deriv', [obs_indices['x'], obs_indices['vx']]),
            ('deriv', [obs_indices['y'], obs_indices['vy']]),
            ('deriv', [obs_indices['z'], obs_indices['vz']]),
        ]
    elif dynamic_constraint_variant == 'pos_legs':
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

    for constraint in dynamic_constraints:
        constraint_list.append(constraint)               

    n_trials = 50                 
    n_timesteps = 300

    fig_all, ax_all = plt.subplots(min(n_trials, 10), len(diffusion_timestep_thresholds), figsize=(20, 20))
    fig_all.suptitle(f'{exp} - dyn. const. {dynamic_constraint_variant}')
    ax_limits = [-6, 6]

    for threshold_index, diff_timestep_threshold in enumerate(diffusion_timestep_thresholds):
        print(f'------------------------Running {exp} - dyn. const.: {dynamic_constraint_variant} - diff time threshold={diff_timestep_threshold}----------------------------')

        minari_dataset = minari.load_dataset(exp, download=True)
        env = minari_dataset.recover_environment()    # Set render_mode='human' to visualize the environment

        dt = 0.05
        projector = Projector(
            horizon=args.horizon, 
            transition_dim=trajectory_dim, 
            constraint_list=constraint_list, 
            normalizer=dataset.normalizer, 
            diffusion_timestep_threshold=diff_timestep_threshold,
            dt=dt,
            cost_dims=cost_dims,
        )

        projector = None if diff_timestep_threshold == -1 else projector
        # Create policy
        policy = Policy(
            model=diffusion,
            scheduler=scheduler,
            normalizer=dataset.normalizer,
            preprocess_fns=args.preprocess_fns,
            test_ret=args.test_ret,
            projector=projector,
        )    

        env.env.env.env.ant_env.frame_skip = 5

        # Run policy
        fig, ax = plt.subplots(min(n_trials, 10), 6, figsize=(20, 20))
        fig.suptitle(f'{exp} - dyn. const. {dynamic_constraint_variant}, diff timestep threshold={diff_timestep_threshold}')

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
            obs, _ = env.reset(seed=i)
            obs = np.concatenate((obs['achieved_goal'], obs['observation'], obs['desired_goal']))
            obs_buffer = []
            action_buffer = []

            sampled_trajectories = []
            disable_projection = True
            for _ in range(n_timesteps):
                conditions = {0: obs}

                # Check if a safety constraint is violated
                for constraint in constraint_list_safe:
                    c, d = constraint[1]
                    if obs @ c >= d + 1e-2:   # (Close to) Violation of constraint
                        n_violations += 1
                        total_violations += obs @ c - d
                        break
                
                start = time.time()
                action, samples = policy(conditions, batch_size=args.batch_size, horizon=args.horizon, disable_projection=disable_projection)
                avg_time[i] += time.time() - start

                # Check whether one of the sampled trajectories violates a 
                disable_projection = True
                for constraint in constraint_list_safe:
                    c, d = constraint[1]
                    if np.any(samples.observations @ c >= d - 1e-2):   # (Close to) Violation of constraint
                        disable_projection = False
                        # print('Enabled projection at timestep', _)
                        break

                if _ % save_samples_every == 0:
                    sampled_trajectories.append(samples.observations[:, :, :])

                obs, rew, terminated, truncated, info = env.step(action)

                dist_to_goal = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
                obs = np.concatenate((obs['achieved_goal'], obs['observation'], obs['desired_goal']))

                # For ant robot, check if it has flipped over or reached the goal (not provided by the environment)
                if obs[obs_indices['z']] < 0.3:    # Ant has likely flipped over
                    terminated = True
                if _ >= 20 and np.linalg.norm(np.array(obs_buffer)[-20:, [obs_indices['vx'], obs_indices['vy']]], axis=1).max() < 0.5:    # Ant is stuck
                    terminated = True
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

            if i >= 10:     # Plot only the first 5 trials
                continue
            plot_states = ['x', 'y', 'vx', 'vy']

            for j in range(len(plot_states)):
                ax[i, j].plot(np.array(obs_buffer)[:, obs_indices[plot_states[j]]])
                ax[i, j].set_title(['x', 'y', 'vx', 'vy'][j])
            
            axes = [ax[i, 4], ax_all[i, threshold_index]]
            for curr_ax in axes:
                curr_ax.plot(np.array(obs_buffer)[:, obs_indices['x']], np.array(obs_buffer)[:, obs_indices['y']], 'k')
                curr_ax.plot(np.array(obs_buffer)[0, obs_indices['x']], np.array(obs_buffer)[0, obs_indices['y']], 'go', label='Start')            # Start
                curr_ax.plot(np.array(obs_buffer)[0, obs_indices['goal_x']], np.array(obs_buffer)[0, obs_indices['goal_y']], 'ro', label='Goal')   # Goal
                curr_ax.set_xlim(ax_limits)
                curr_ax.set_ylim(ax_limits)
            
            axes = [ax[i, 5], ax_all[i, threshold_index]]
            for __ in range(len(sampled_trajectories_all[i])):          # Iterate over timesteps of sampled trajectories
                for ___ in range(min(args.batch_size, 4)):              # Iterate over batch
                    close_to_origin_threshold = 1
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
            axes = [ax[i, 4], ax[i, 5], ax_all[i, threshold_index]]
            for curr_ax in axes:
                curr_ax.add_patch(matplotlib.patches.Rectangle((-6, -2), 8, 4, color='k', alpha=0.2))

                for constraint in safety_constraints:
                    mat = np.zeros((3, 2))
                    mat[:2] = constraint[:2]
                    mat[2] = np.array([6, -6]) if constraint[2] == 'above' else np.array([6, 6])
                    curr_ax.add_patch(matplotlib.patches.Polygon(mat, color='c', alpha=0.2))

        print(f'Success rate: {n_success / n_trials}')
        if n_success > 0:
            print(f'Avg number of steps: {n_steps / n_trials}')
            print(f'Avg number of constraint violations: {n_violations / n_trials}')
            print(f'Avg total violation: {total_violations / n_trials}')
        print(f'Average computation time per step: {np.mean(avg_time)}')

        fig.savefig(f'{args.savepath}/diff_timestep_th/dyn_const_{dynamic_constraint_variant}_th_{diff_timestep_threshold}.png')   

        ax_all[0, threshold_index].set_title(f'Diff. time threshold: {diff_timestep_threshold}')
        env.close()

    fig_all.savefig(f'{args.savepath}/diff_timestep_th/dyn_const_{dynamic_constraint_variant}_all.png')
    plt.show()