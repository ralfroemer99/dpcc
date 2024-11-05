import yaml
import numpy as np
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
    print(f'${steps_avg:.1f} \pm {steps_std:.1f}$ & ${success_rate_goal:.2f}$ & ${success_rate_constraints:.2f}$ & ${n_violations_avg:.1f} \pm {n_violations_std:.1f}$ & ${total_violations_avg:.2f} \pm {total_violations_std:.2f}$ \\\\')
