import yaml
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import diffuser.utils as utils
from diffuser.sampling import Policy
from d3il.environments.d3il.envs.gym_avoiding_env.gym_avoiding.envs.avoiding import ObstacleAvoidanceEnv

exp = 'avoiding-d3il'

# Load configuration
with open('config/projection_eval.yaml', 'r') as file:
    config = yaml.safe_load(file)

obs_indices = config['observation_indices']['avoiding']
act_indices = config['action_indices']['avoiding']

class Parser(utils.Parser):
    dataset: str = exp
    config: str = 'config.' + exp

args = Parser().parse_args(experiment='plan')

# Get model
diffusion_experiment = utils.load_diffusion(args.loadbase, args.dataset, args.diffusion_loadpath, str(args.seed), epoch=args.diffusion_epoch, device=args.device)
diffusion = diffusion_experiment.diffusion
dataset = diffusion_experiment.dataset

# Create policy
policy = Policy(model=diffusion, normalizer=dataset.normalizer, preprocess_fns=args.preprocess_fns, test_ret=args.test_ret) 

# -------------------- Run experiments ------------------
env = ObstacleAvoidanceEnv(render=False)
env.start()

ax_limits = config['ax_limits'][exp]

errors = np.zeros((10, 100))

for i in range(2):
    torch.manual_seed(i)
    # Reset environment
    obs = env.reset()
    action = env.robot_state()[:2]
    fixed_z = env.robot_state()[2:]

    obs = np.concatenate((action[:2], obs))      
        
    terminated = False

    obs_buffer = []
    action_buffer = []

    t = 0
    while not terminated and t < 100:
        # Compute action
        action, samples = policy(conditions={0: obs}, batch_size=args.batch_size, horizon=args.horizon)           

        obs_buffer.append(obs)
        action_buffer.append(action)

        # Step environment
        next_pos_des = action + obs[:2] 
        obs, rew, terminated, info = env.step(np.concatenate((next_pos_des, fixed_z, [0, 1, 0, 0]), axis=0))
        success = info[1]

        obs = np.concatenate((next_pos_des[:2], obs))

        # Get error in the position
        pos_current = obs[2:4]
        errors[i, t] = np.linalg.norm(pos_current - next_pos_des[:2])

print(f'Mean error: {np.mean(errors[errors > 0]):.4f}')
print(f'Max error: {np.max(errors):.4f}')

obs_all = np.array(obs_buffer)
action_all = np.array(action_buffer)

# Predict state using different dynamics models
start_at = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36]
horizon = 21

fig1, ax1 = plt.subplots(2, len(start_at), figsize=(20, 5))
fig2, ax2 = plt.subplots(1, 1, figsize=(4, 5))
fig3, ax3 = plt.subplots(1, 2, figsize=(15, 3))
labels = ['x_des', 'y_des', 'x', 'y']
for ax_idx, t_start in enumerate(start_at):
    obs_predict = np.zeros((2, horizon, obs_all.shape[1]))
    obs_predict[0] = obs_all[t_start:t_start + horizon]
    obs_predict[1, 0] = obs_predict[0, 0]

    for t in np.arange(1, horizon):
        obs_predict[1, t, 0] = obs_predict[1, t - 1, 0] + action_all[t_start + t - 1, 0]      # x_des
        obs_predict[1, t, 1] = obs_predict[1, t - 1, 1] + action_all[t_start + t - 1, 1]      # y_des
        obs_predict[1, t, 2] = obs_predict[1, t - 1, 2] + action_all[t_start + t - 1, 0]      # x
        obs_predict[1, t, 3] = obs_predict[1, t - 1, 3] + action_all[t_start + t - 1, 1]      # y

    # -------------------- Plot results ------------------
    for i in np.arange(2, 4):
        ax1[i - 2, ax_idx].plot(obs_predict[0, :, i], 'r', label='observed')
        ax1[i - 2, ax_idx].plot(obs_predict[1, :, i], 'b', label='predicted')
        ax1[i - 2, ax_idx].legend()
        ax3[i - 2].plot(obs_predict[0, :, i] - obs_predict[1, :, i])
    ax2.plot(obs_predict[0, :, 2], obs_predict[0, :, 3], 'r', label='predicted')

ax1[0, 0].set_ylabel('$x$', fontsize=16)
ax1[1, 0].set_ylabel('$y$', fontsize=16)
for i in range(5):
    ax1[1, i].set_xlabel('$t$', fontsize=16)

ax2.plot(obs_all[:, 2], obs_all[:, 3], 'b', label='observed')

ax3[0].set_ylabel('$x$ error', fontsize=16)
ax3[1].set_ylabel('$y$ error', fontsize=16)
for i in range(2):
    ax3[i].set_xlabel('$t$', fontsize=16)
    ax3[i].set_ylim(-0.035, 0.035)
    ax3[i].set_xlim(0, horizon)
    ax3[i].set_xticks([0, 5, 10, 15, 20])
    ax3[i].set_xticklabels(['0', '5', '10', '15', '20'], fontsize=16)
    ax3[i].set_yticks([-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03])
    ax3[i].set_yticklabels(['-0.03', '-0.02', '-0.01', '0', '0.01', '0.02', '0.03'], fontsize=16)

# Adjust layout to avoid collision
fig3.tight_layout()
fig3.subplots_adjust(left=0.1)  # Adjust the left margin to provide more space for the ylabel

save_path = f'{args.savepath}/results/no_obstacles'
os.makedirs(save_path, exist_ok=True)
fig1.savefig(f'{save_path}/obs_actions.pdf', bbox_inches='tight', format='pdf')
fig2.savefig(f'{save_path}/trajectories.png')
fig3.savefig(f'{save_path}/errors.pdf', bbox_inches='tight', format='pdf')
fig3.savefig(f'{save_path}/errors.png', bbox_inches='tight')
plt.show()