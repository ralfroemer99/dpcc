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

# Create trainer
trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    noise_scheduler=DDPMScheduler(num_train_timesteps=args.n_diffusion_steps),
    train_test_split=args.train_test_split,
    loss_type=args.loss_type,
    action_weight=args.action_weight,
    ema_decay=args.ema_decay,
    n_train_steps=args.n_train_steps,
    n_steps_per_epoch=args.n_steps_per_epoch,
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    results_folder=args.savepath,
)

trainer = trainer_config(model=model, dataset=dataset)

trainer.train()
