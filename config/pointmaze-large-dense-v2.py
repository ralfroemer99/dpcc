import socket

from diffuser.utils import watch

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('returns_condition', 'R'),
    ## value kwargs
    # ('discount', 'd'),
]

logbase = 'logs'

base = {
    'diffusion': {
        ## model
        'model': 'models.UNet1DTemporalCondModel',
        'diffusion': 'models.GaussianInvDynDiffusion',
        'horizon': 16,
        'n_diffusion_steps': 20,
        'loss_type': 'l2',
        'loss_discount': 1.0,
        'returns_condition': False,
        'action_weight': 10,            
        'dim': 32,
        'dim_mults': (1, 2, 4, 8),
        'hidden_dim': 256,
        'attention': False,
        'condition_dropout': 0.25,
        'condition_guidance_w': 1.2,
        'test_ret': 0.9,        

        ## dataset
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 450,
        'include_returns': True,
        'returns_scale': 400,   # Determined using rewards from the dataset
        'discount': 0.99,

        ## serialization
        'logbase': logbase,
        'prefix': 'diffusion/defaults',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 1000,  # 10000
        'n_train_steps': 1e5,       # 1e6
        'batch_size': 32,            # 32
        'learning_rate': 1e-4,      # 2e-4
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'train_test_split': 0.9,
        'device': 'cuda',
        'seed': 0,
    },

    'plan': {
        'policy': 'sampling.Policy',
        'max_episode_length': 100,
        'batch_size': 16,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': 0,
        'test_ret': 0,

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/',
        'exp_name': watch(args_to_watch),

        ## diffusion model
        'horizon': 16,
        'n_diffusion_steps': 20,
        'returns_condition': False,
        'use_padding': True,

        ## loading
        'diffusion_loadpath': 'f:diffusion/defaults_H{horizon}_T{n_diffusion_steps}_R{returns_condition}',
        'value_loadpath': 'f:values/defaults_H{horizon}_T{n_diffusion_steps}',

        'diffusion_epoch': 'latest',      # 'latest'

        'verbose': False,
        'suffix': '0',
    },
}


#------------------------ overrides ------------------------#

halfcheetah_medium_replay_v2 = halfcheetah_medium_v2 = halfcheetah_medium_expert_v2 = {
    'diffusion': {
        'horizon': 4,
        'dim_mults': (1, 4, 8),
        'attention': True,
    },
    'values': {
        'horizon': 4,
        'dim_mults': (1, 4, 8),
    },
    'plan': {
        'horizon': 4,
        'scale': 0.001,
        't_stopgrad': 4,
    },
}