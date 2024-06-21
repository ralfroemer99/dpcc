import time
import minari
import numpy as np
import diffuser.utils as utils
import matplotlib.pyplot as plt
from diffuser.sampling import Policy
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

exp = 'pointmaze-umaze-dense-v2'

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
diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
trainer = diffusion_experiment.trainer

model = diffusion

# Create scheduler
scheduler = DDPMScheduler(num_train_timesteps=trainer.noise_scheduler.config.num_train_timesteps)   
scheduler.set_timesteps(20)                         # Steps used for inference

# Create policy
policy = Policy(
    model=model,
    scheduler=scheduler,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
)    

# Plot losses
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
utils.plot_losses(diffusion_losses, ax=ax[0], title='Diffusion losses')
# utils.plot_losses(value_losses, ax=ax[1], title='Value losses')
plt.show()

dataset = minari.load_dataset(exp, download=True)
env = dataset.recover_environment(render_mode='human', eval_env=True)

# Run policy
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
        conditions = {0: np.concatenate((obs['observation'], obs['desired_goal']))}
        start = time.time()
        action, samples = policy(conditions, batch_size=args.batch_size, horizon=args.horizon)
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

input("Press Enter to close the window...")
env.close()