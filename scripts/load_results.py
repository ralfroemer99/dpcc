import yaml
import numpy as np
import matplotlib.pyplot as plt
import diffuser.utils as utils

# Load configuration
with open('config/projection_eval.yaml', 'r') as file:
    config = yaml.safe_load(file)

projection_variants = config['projection_variants']

exp = 'avoiding-d3il'
class Parser(utils.Parser):
    dataset: str = exp
    config: str = 'config.' + exp

seeds = config['seeds']
avoiding_halfspace_variants = config['avoiding_halfspace_variants']

sr_goal_all = {}
sr_constraints_all = {}
timesteps_avg_all = {}
timesteps_std_all = {}

for variant in projection_variants:
    n_success_all = np.array([])
    n_success_and_constraints_all = np.array([])
    n_steps_all = np.array([])
    n_violations_all = np.array([])
    total_violations_all = np.array([])
    collision_free_completed_all = np.array([])
    for halfspace_variant in avoiding_halfspace_variants:
        for i, seed in enumerate(seeds):
            args = Parser().parse_args(experiment='plan', seed=seed)

            # Get data
            data = np.load(f'{args.savepath}/results/halfspace_{halfspace_variant}/{variant}.npz', allow_pickle=True)
            n_success = data["n_success"]
            n_success_and_constraints = data["n_success_and_constraints"]
            n_steps = data["n_steps"]
            n_violations = data["n_violations"]
            total_violations = data["total_violations"]
            avg_time = data["avg_time"]
            collision_free_completed = data["collision_free_completed"]

            n_success_all = np.append(n_success_all, n_success)
            n_success_and_constraints_all = np.append(n_success_and_constraints_all, n_success_and_constraints)
            n_steps_all = np.append(n_steps_all, n_steps[n_success > 0])
            n_violations_all = np.append(n_violations_all, n_violations)
            total_violations_all = np.append(total_violations_all, total_violations)
            collision_free_completed_all = np.append(collision_free_completed_all, collision_free_completed)

    success_rate_goal = n_success_all.mean()
    success_rate_goal_constraints = n_success_and_constraints_all.mean()
    success_rate_constraints = collision_free_completed_all.mean()
    steps_avg = n_steps_all.mean()
    steps_std = n_steps_all.std()
    n_violations_avg = n_violations_all.mean()
    n_violations_std = n_violations_all.std()
    total_violations_avg = total_violations_all.mean()
    total_violations_std = total_violations_all.std()

    print(f'------------------ Variant: {variant} ------------------')
    print(f'Success rate (goal): {success_rate_goal:.2f}')
    print(f'Success rate (goal + constraints): {success_rate_goal_constraints:.2f}')
    print(f'Success rate (constraints): {success_rate_constraints:.2f}')
    print(f'Average steps: {steps_avg:.2f} +- {steps_std:.2f}')
    print(f'Average violations: {n_violations_avg:.2f} +- {n_violations_std:.2f}')
    print(f'Average total violations: {total_violations_avg:.3f} +- {total_violations_std:.3f}')
    print(f'Average time: {avg_time.mean():.2f} +- {avg_time.std():.2f}')
    print(f'${steps_avg:.1f} \pm {steps_std:.1f}$ & ${success_rate_goal:.2f}$ & ${success_rate_constraints:.2f}$ & ${n_violations_avg:.1f} \pm {n_violations_std:.1f}$ \\\\')

    sr_goal_all[variant] = success_rate_goal
    sr_constraints_all[variant] = success_rate_constraints
    timesteps_avg_all[variant] = steps_avg
    timesteps_std_all[variant] = steps_std

# Plot results
variants_to_plot = ['ours-random-project_x_t', 'ours-consistency-project_x_t', 'ours-costs-project_x_t']
variants_labels = ['DPCC-R', 'DPCC-T', 'DPCC-C']
# variants_to_plot = ['ours-enlarged-random-project_x_t', 'ours-enlarged-consistency-project_x_t', 'ours-enlarged-costs-project_x_t']
# variants_labels = ['DPCC-RT', 'DPCC-TT', 'DPCC-CT']

# Extract success rates for each variant
sr_goal = [sr_goal_all[variant] for variant in variants_to_plot]
sr_constraints = [sr_constraints_all[variant] for variant in variants_to_plot]
timesteps_avg = [timesteps_avg_all[variant] for variant in variants_to_plot]
timesteps_std = [timesteps_std_all[variant] for variant in variants_to_plot]
print(sr_goal)
print(sr_constraints)
print(timesteps_avg)
print(timesteps_std)

# Create a bar plot
x = np.arange(len(variants_to_plot))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 10))
bars1 = ax.bar(x - width/2, sr_goal, width, label='Goal reached', color='green')
bars2 = ax.bar(x + width/2, sr_constraints, width, label='Constraints satisfied', color='red')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Success Rate', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(variants_labels, fontsize=12)
plt.setp(ax.get_yticklabels(), fontsize=12)
ax.legend(loc='lower left', fontsize=12) 

# Add labels to the bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
add_labels(bars1)
add_labels(bars2)

fig.tight_layout()

plt.savefig('success_rates.png')
plt.show()

# Create the second bar plot for timesteps
fig, ax = plt.subplots(figsize=(10, 10))
bars = ax.bar(x, timesteps_avg, width, yerr=timesteps_std, label='Timesteps', color=[0.5, 0.5, 1], capsize=5)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xticks(x)
ax.set_xticklabels(variants_labels, fontsize=12)
plt.setp(ax.get_yticklabels(), fontsize=12)
ax.set_ylim([0, 100])
ax.legend(loc='lower left', fontsize=12) 

# Add labels to the bars
add_labels(bars)

fig.tight_layout()
plt.savefig('timesteps.png')
plt.show()
