import diffuser.utils as utils
from diffuser.sampling import Projector

exps = [
        # 'pointmaze-open-dense-v2',
        'pointmaze-umaze-dense-v2',
        # 'pointmaze-medium-dense-v2',
        # 'pointmaze-large-dense-v2',
        # 'antmaze-umaze-v1',
        # 'relocate-cloned-v2',
        ]

for exp in exps:
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
        include_returns=args.include_returns,
        # returns_scale=args.returns_scale,
        returns_scale=args.max_path_length,               # Because the reward is <= 1 in each timestep
        # returns_scale=args.n_diffusion_steps,
        discount=args.discount,
    )

    dataset = dataset_config()
    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim
    goal_dim = dataset.goal_dim


    # -----------------------------------------------------------------------------#
    # --------------------------- dynamics constraints ----------------------------#
    # -----------------------------------------------------------------------------#

    if 'pointmaze' in exp:
        obs_indices = {'x': 0, 'y': 1, 'vx': 2, 'vy': 3, 'goal_x': 4, 'goal_y': 5}
    elif 'antmaze' in exp:
        obs_indices = {'x': 0, 'y': 1, 'z':2, 'qx': 3, 'qy': 4, 'qz': 5, 'qw': 6, 'hip1': 7, 'ankle1': 8, 'hip2': 9, 'ankle2': 10, 
                       'hip3': 11, 'ankle3': 12, 'hip4': 13, 'ankle4': 14, 'vx': 15, 'vy': 16, 'vz': 17, 'dhip1': 21, 'dankle1': 22,
                       'dhip2': 23, 'dankle2': 24, 'dhip3': 25, 'dankle3': 26, 'dhip4': 27, 'dankle4': 28, 'goal_x': 29, 'goal_y': 30, }

    if 'pointmaze' in exp:
        dynamic_constraints = [
            ('deriv', [obs_indices['x'], obs_indices['vx']]),
            ('deriv', [obs_indices['y'], obs_indices['vy']]),
        ]
    elif 'antmaze' in exp:
        dynamic_constraints = [
            ('deriv', [obs_indices['x'], obs_indices['vx']]),
            ('deriv', [obs_indices['y'], obs_indices['vy']]),
            ('deriv', [obs_indices['z'], obs_indices['vz']]),
            # ('deriv', [obs_indices['hip1'], obs_indices['dhip1']]),
            # ('deriv', [obs_indices['ankle1'], obs_indices['dankle1']]),
            # ('deriv', [obs_indices['hip2'], obs_indices['dhip2']]),
            # ('deriv', [obs_indices['ankle2'], obs_indices['dankle2']]),
            # ('deriv', [obs_indices['hip3'], obs_indices['dhip3']]),
            # ('deriv', [obs_indices['ankle3'], obs_indices['dankle3']]),
            # ('deriv', [obs_indices['hip4'], obs_indices['dhip4']]),
            # ('deriv', [obs_indices['ankle4'], obs_indices['dankle4']]),
        ]
    
    trajectory_dim = observation_dim + action_dim if args.model == 'models.GaussianDiffusion' else observation_dim
    dt = 0.02 if 'pointmaze' in exp else 0.05

    # Create projector
    projector = Projector(
            horizon=args.horizon, 
            transition_dim=trajectory_dim, 
            constraint_list=dynamic_constraints, 
            normalizer=dataset.normalizer, 
            dt=dt,
            skip_initial_state=False,
        )

    # -----------------------------------------------------------------------------#
    # ------------------------------ model & trainer ------------------------------#
    # -----------------------------------------------------------------------------#

    if args.diffusion == 'models.GaussianInvDynDiffusion':
        model_config = utils.Config(
            args.model,
            savepath=(args.savepath, 'model_config.pkl'),
            horizon=args.horizon,
            transition_dim=observation_dim,
            cond_dim=observation_dim,
            dim_mults=args.dim_mults,
            returns_condition=args.returns_condition,
            dim=args.dim,
            condition_dropout=args.condition_dropout,
            device=args.device,
        )

        diffusion_config = utils.Config(
            args.diffusion,
            savepath=(args.savepath, 'diffusion_config.pkl'),
            horizon=args.horizon,
            observation_dim=observation_dim,
            action_dim=action_dim,
            goal_dim=dataset.goal_dim,
            n_timesteps=args.n_diffusion_steps,
            loss_type=args.loss_type,
            clip_denoised=args.clip_denoised,
            predict_epsilon=args.predict_epsilon,
            hidden_dim=args.hidden_dim,
            ## loss weighting
            dynamics_loss=args.dynamic_loss,
            projector=projector,
            action_weight=args.action_weight,
            loss_discount=args.loss_discount,
            returns_condition=args.returns_condition,
            condition_guidance_w=args.condition_guidance_w,
            device=args.device,
        )
    else:
        model_config = utils.Config(
            args.model,
            savepath=(args.savepath, 'model_config.pkl'),
            horizon=args.horizon,
            transition_dim=observation_dim + action_dim,
            cond_dim=observation_dim,
            dim_mults=args.dim_mults,
            returns_condition=args.returns_condition,
            dim=args.dim,
            condition_dropout=args.condition_dropout,
            device=args.device,
        )

        diffusion_config = utils.Config(
            args.diffusion,
            savepath=(args.savepath, 'diffusion_config.pkl'),
            horizon=args.horizon,
            observation_dim=observation_dim,
            action_dim=action_dim,
            n_timesteps=args.n_diffusion_steps,
            loss_type=args.loss_type,
            clip_denoised=args.clip_denoised,
            ## loss weighting
            action_weight=args.action_weight,
            loss_discount=args.loss_discount,
            returns_condition=args.returns_condition,
            condition_guidance_w=args.condition_guidance_w,
            device=args.device,
        )

    # Create trainer
    trainer_config = utils.Config(
        utils.Trainer,
        savepath=(args.savepath, 'trainer_config.pkl'),
        train_test_split=args.train_test_split,
        ema_decay=args.ema_decay,
        n_train_steps=args.n_train_steps,
        n_steps_per_epoch=args.n_steps_per_epoch,
        train_batch_size=args.batch_size,
        train_lr=args.learning_rate,
        gradient_accumulate_every=args.gradient_accumulate_every,
        results_folder=args.savepath,
    )

    # -----------------------------------------------------------------------------#
    # -------------------------------- instantiate --------------------------------#
    # -----------------------------------------------------------------------------#
    model = model_config()
    diffusion = diffusion_config(model)
    trainer = trainer_config(diffusion, dataset)

    trainer.train()
