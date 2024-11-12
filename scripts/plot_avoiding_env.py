import yaml
import matplotlib
import matplotlib.pyplot as plt
import diffuser.utils as utils

# Load configuration
with open('config/projection_eval.yaml', 'r') as file:
    config = yaml.safe_load(file)

# General
exp = 'avoiding-d3il'
halfspace_variants = config['avoiding_halfspace_variants']
ax_limits = config['ax_limits'][exp]

# Constraint projection
repeat_last = config['repeat_last']
diffusion_timestep_threshold = config['diffusion_timestep_threshold']
constraint_types = config['constraint_types']

fig, axes = plt.subplots(1, 3, figsize=(30, 10))

for i, ax in enumerate(axes):
    if halfspace_variants[i] == 'top-left':
        polytopic_constraints = [config['halfspace_constraints'][exp][0]]
        obstacle_constraints = [config['obstacle_constraints'][exp][0]]
    elif halfspace_variants[i] == 'top-right':
        polytopic_constraints = [config['halfspace_constraints'][exp][1]]
        obstacle_constraints = [config['obstacle_constraints'][exp][1]]
    else:
        polytopic_constraints = [config['halfspace_constraints'][exp][2], config['halfspace_constraints'][exp][3]]
        obstacle_constraints = [config['obstacle_constraints'][exp][2]]
    utils.plot_environment_constraints(exp, ax)
    utils.plot_halfspace_constraints(exp, polytopic_constraints, ax, ax_limits)
    for constraint in obstacle_constraints:
        ax.add_patch(matplotlib.patches.Circle(constraint['center'], constraint['radius'], color='b', alpha=0.2))
    # Add very thick light green line at y = 0.35
    ax.plot([ax_limits[0][0], ax_limits[0][1]], [0.35, 0.35], color=[0.4, 1, 0.4], linewidth=5)
    ax.set_xlim(ax_limits[0])
    ax.set_ylim(ax_limits[1])
    ax.set_facecolor([1, 1, 0.9])

plt.show()
