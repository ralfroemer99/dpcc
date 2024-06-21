from collections import namedtuple
import torch
import time
import einops
import numpy as np
import pdb

import diffuser.utils as utils
from diffuser.datasets.preprocessing import get_policy_preprocess_fn


Trajectories = namedtuple('Trajectories', 'actions observations')


class Policy:

    def __init__(self, model, scheduler, normalizer, preprocess_fns, **sample_kwargs):
        self.model = model
        self.scheduler = scheduler
        self.normalizer = normalizer
        self.action_dim = model.action_dim
        self.preprocess_fn = get_policy_preprocess_fn(preprocess_fns)
        self.sample_kwargs = sample_kwargs

    def __call__(self, conditions, batch_size=1, horizon=16, id_model=None):
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        conditions = self._format_conditions(conditions, batch_size)

        ## run reverse diffusion process
        sample_size = (batch_size, horizon, self.model.observation_dim + self.model.action_dim)
        noise = torch.randn(sample_size, device=self.device)
        sample = noise
        for t in self.scheduler.timesteps:
            with torch.no_grad():
                noisy_residual = self.model(sample=sample, timestep=t, condition=conditions)
            previous_noisy_sample = self.scheduler.step(noisy_residual, t, sample).prev_sample
            sample = previous_noisy_sample

        # samples = self.model(conditions, guide=self.guide, verbose=verbose, **self.sample_kwargs)
        trajectories = utils.to_np(sample)

        ## extract observations [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')

        ## extract action [ batch_size x horizon x action_dim ]
        if self.action_dim > 0:
            actions = trajectories[:, :, :self.action_dim]
            actions = self.normalizer.unnormalize(actions, 'actions')

            ## extract first action
            action = actions[0, 0]
        else:
            actions = None
            if id_model is not None:
                with torch.no_grad():
                    obs = normed_observations[0, 0]
                    next_obs = normed_observations[0, 1]
                    normed_action = id_model(torch.tensor(obs).float(), torch.tensor(next_obs).float()).detach().numpy()
                    action = self.normalizer.unnormalize(normed_action, 'actions')
            else:
                action = None

        trajectories = Trajectories(actions, observations)

        # self.previous_trajectories = samples.trajectories

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
    