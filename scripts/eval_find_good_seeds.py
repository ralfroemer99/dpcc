import minari
import diffuser.utils as utils


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

dataset = diffusion_experiment.dataset

dataset = minari.load_dataset(exp, download=True)
env = dataset.recover_environment(eval_env=True) if 'pointmaze' in exp else dataset.recover_environment()     # Set render_mode='human' to visualize the environment

if 'pointmaze' in exp:
    env.env.env.env.point_env.frame_skip = 2
if 'antmaze' in exp:
    env.env.env.env.ant_env.frame_skip = 5

# Run policy
n_trials = 100

# Store a few sampled trajectories
if 'pointmaze' in exp:
    obs_indices = {'x': 0, 'y': 1, 'vx': 2, 'vy': 3, 'goal_x': 4, 'goal_y': 5}
elif 'antmaze' in exp:
    obs_indices = {'x': 0, 'y': 1, 'z':2, 'vx': 15, 'vy': 16, 'vz': 17, 'goal_x': 29, 'goal_y': 30, 'qx': 3, 'qy': 4, 'qz': 5, 'qw': 6}

good_seeds = []
for i in range(n_trials):
    obs, _ = env.reset(seed=i)
    
    x0 = obs['observation'][obs_indices['x']]
    y0 = obs['observation'][obs_indices['y']]
    if x0 <= 0.25 and y0 <= -0.5:
        good_seeds.append(i)
    
print(good_seeds)
env.close()