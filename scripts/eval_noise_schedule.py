import time
import minari
import numpy as np
import scipy as sp
import diffuser.utils as utils
import matplotlib.pyplot as plt
from diffuser.sampling import Policy
from diffusers import DDPMScheduler, DDIMScheduler
from diffuser.sampling import Projector


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
diffusion = diffusion_experiment.diffusion
dataset = diffusion_experiment.dataset

# Get trajectory from dataset


# Add noise


# Plot constraint violations

