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

seeds = [0, 1, 2]

for variant in projection_variants:
    n_success_all = np.array([])
    n_steps_all = np.array([])
    n_violations_all = np.array([])
    total_violations_all = np.array([])
    collision_free_completed_all = np.array([])
    for i, seed in enumerate(seeds):
        args = Parser().parse_args(experiment='plan', seed=seed)

        # Get data
        data = np.load(f'{args.savepath}/results/{variant}.npz', allow_pickle=True)
        n_success = data["n_success"]
        n_steps = data["n_steps"]
        n_violations = data["n_violations"]
        total_violations = data["total_violations"]
        avg_time = data["avg_time"]
        collision_free_completed = data["collision_free_completed"]

        n_success_all = np.append(n_success_all, n_success)
        n_steps_all = np.append(n_steps_all, n_steps)
        n_violations_all = np.append(n_violations_all, n_violations)
        total_violations_all = np.append(total_violations_all, total_violations)
        collision_free_completed_all = np.append(collision_free_completed_all, collision_free_completed)

    print(f'Variant: {variant}')
    print(f'Success rate (goal): {n_success_all.mean():.3f}')
    print(f'Success rate (constraints): {collision_free_completed_all.mean():.3f}')
    print(f'Average steps: {n_steps_all.mean():.2f} +- {n_steps_all.std():.2f}')
    print(f'Average violations: {n_violations_all.mean():.2f} +- {n_violations_all.std():.2f}')
    print(f'Average total violations: {total_violations_all.mean():.3f} +- {total_violations_all.std():.3f}')
