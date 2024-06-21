import socket

from diffuser.utils import watch

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('use_actions', 'A'),
    ## value kwargs
    ('discount', 'd'),
]

logbase = 'logs'

base = {
    'diffusion': {
        ## model
        'model': 'models.UNet1DTemporalModel',
        'horizon': 16,
        'n_diffusion_steps': 20,
        'loss_type': 'l2',
        'action_weight': 10,            # To include 
        'dim_mults': (1, 2, 4, 8),
        'attention': False,
        # 'renderer': 'utils.MuJoCoRenderer',

        ## dataset
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': False,
        'max_path_length': 100,

        ## serialization
        'logbase': logbase,
        'prefix': 'diffusion/defaults',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 1000,  # 10000
        'n_train_steps': 1e6,       # 1e6
        'batch_size': 32,            # 32
        'learning_rate': 2e-5,      # 2e-4
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

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/',
        'exp_name': watch(args_to_watch),

        ## diffusion model
        'horizon': 16,
        'n_diffusion_steps': 20,

        ## loading
        'diffusion_loadpath': 'f:diffusion/defaults_H{horizon}_T{n_diffusion_steps}',
        'value_loadpath': 'f:values/defaults_H{horizon}_T{n_diffusion_steps}',

        'diffusion_epoch': 'best',      # 'latest'

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