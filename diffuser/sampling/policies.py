from collections import namedtuple
import torch
import time
import einops
import numpy as np
import diffuser.utils as utils
from diffusers.pipelines import DiffusionPipeline
from diffuser.datasets.preprocessing import get_policy_preprocess_fn
from diffuser.models.helpers import apply_conditioning
from diffuser.utils.arrays import to_device


Trajectories = namedtuple('Trajectories', 'actions observations')


class Policy:

    def __init__(self, model, scheduler, normalizer, preprocess_fns=[], test_ret=0, projector=None, **sample_kwargs):
        self.model = model
        self.scheduler = scheduler,   # 'DDPM' or 'DDIM'
        self.scheduler = self.scheduler[0]      # No idea why this is needed
        self.normalizer = normalizer
        self.action_dim = model.action_dim
        self.preprocess_fn = get_policy_preprocess_fn(preprocess_fns)
        self.test_ret = test_ret
        # self.return_diffusion = return_diffusion
        self.sample_kwargs = sample_kwargs

        # Inverse dynamics model
        if self.model.__class__.__name__ == 'GaussianInvDynDiffusion':
            self.inverse_dynamics = True
            self.inv_model = self.model.inv_model
            self.action_dim = 0
        else:
            self.inverse_dynamics = False

        # Projector
        self.projector = projector

    def __call__(self, conditions, batch_size=1, horizon=16, test_ret=None, constraints=None, disable_projection=False):
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        conditions = self._format_conditions(conditions, batch_size)

        test_ret = test_ret if test_ret is not None else self.test_ret
        returns = to_device(test_ret * torch.ones(batch_size, 1), 'cuda')

        # Use GaussianDiffusion model with DDPM
        projector = self.projector if not disable_projection else None
        samples, infos = self.model(conditions, returns=returns, projector=projector, constraints=constraints, **self.sample_kwargs)

        # if self.return_diffusion:
        #     samples, diffusion = self.model(conditions, returns=returns, projector=projector, constraints=constraints, return_diffusion=True, **self.sample_kwargs)
        # else:
        #     samples = self.model(conditions, returns=returns, projector=projector, constraints=constraints, **self.sample_kwargs)

        # Use UNet with variable scheduler
        # shape = (batch_size, horizon, self.model.observation_dim + self.action_dim)
        # noise = 0.5 * torch.randn(shape, device=self.device)
        # samples = noise
        # samples = apply_conditioning(samples, conditions, self.action_dim, self.model.goal_dim)
        # for t in self.scheduler.timesteps:
        #     with torch.no_grad():
        #         epsilon_cond = self.model.model(x=samples, cond=conditions, time=t, returns=returns, use_dropout=False)
        #         epsilon_uncond = self.model.model(x=samples, cond=conditions, time=t, returns=to_device(0 * torch.ones(batch_size, 1), 'cuda'), use_dropout=True)
        #         noisy_residual = epsilon_uncond + self.model.condition_guidance_w * (epsilon_cond - epsilon_uncond)     # Predict noise epsilon
        #     previous_noisy_sample = self.scheduler.step(noisy_residual, t, samples).prev_sample
        #     samples = previous_noisy_sample
        #     samples = apply_conditioning(samples, conditions, self.action_dim, self.model.goal_dim)

        trajectories = utils.to_np(samples)

        if self.inverse_dynamics:
            obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)
            obs_comb = obs_comb.reshape(-1, 2*samples.shape[-1])
            actions = self.inv_model(obs_comb)
            actions = utils.to_np(actions)
            actions = self.normalizer.unnormalize(actions, 'actions')
            if not 'projection_costs' in infos or infos['projection_costs'] == {}:
                action = actions[0]     # Change this to follow "safest" trajectory
            else:
                costs_total = np.zeros(batch_size)
                for timestep, cost in infos['projection_costs'].items():
                    costs_total += cost
                action = actions[np.argmin(costs_total)]
        else:
            ## extract action [ batch_size x horizon x action_dim ]
            actions = trajectories[:, :, :self.action_dim]
            actions = self.normalizer.unnormalize(actions, 'actions')

            ## extract first action
            action = actions[0, 0]

        ## extract observations [ batch_size x horizon x observation_dim ]
        if not 'diffusion' in infos:
            normed_observations = trajectories[:, :, self.action_dim:]
            observations = self.normalizer.unnormalize(normed_observations, 'observations')
        if 'diffusion' in infos:
            diffusion_trajectories = utils.to_np(infos['diffusion'])         # Shape: batch_size x T x horizon x transition_dim     
            observations = self.normalizer.unnormalize(diffusion_trajectories[:, :, :, self.action_dim:], 'observations')
        
        trajectories = Trajectories(actions, observations)

        return action, trajectories

    @property
    def device(self):
        parameters = list(self.model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda:0')
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions
    