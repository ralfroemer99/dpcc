from diffusers import UNet1DModel

network = UNet1DModel.from_pretrained("bglick13/hopper-medium-v2-value-function-hor32", subfolder="unet").to(device='cuda')

# print(network.down_blocks)
# print(network.mid_block)
# print(network.up_blocks)
# print(network.out_block)

import diffuser.utils as utils
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

exp = 'pointmaze-umaze-dense-v2'

class Parser(utils.Parser):
    dataset: str = exp
    config: str = 'config.' + exp

args = Parser().parse_args('diffusion')

# Get dataset
dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'dataset_config.pkl'),
    env=args.dataset,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
)

dataset = dataset_config()

# Create model
model_config = utils.Config(
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),
    horizon=args.horizon,
    observation_dim=dataset.observation_dim,
    action_dim=dataset.action_dim,
    goal_dim=dataset.goal_dim,
    dim_mults=(1, 2, 4, 8),
    attention=args.attention,
    device=args.device,    
)
model = model_config()

print('Hi')