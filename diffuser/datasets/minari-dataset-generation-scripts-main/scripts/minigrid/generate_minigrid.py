import os
import sys
import argparse
import gymnasium as gym
import minigrid
import minari
from minari import DataCollector
from gymnasium.spaces.text import alphanumeric
import numpy as np
np.random.seed(42)

from policies import RandomPolicy, ExpertPolicy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../checks")))
from check_dataset import run_all_checks

parser = argparse.ArgumentParser(description='Generate MiniGrid FourRooms datasets')
parser.add_argument('--random', help='Whether to user random policy', action='store_true')

def generate_dataset(random):
    env = gym.make("MiniGrid-FourRooms-v0")
    max_len = len("reach the goal")
    obs_space = gym.spaces.Dict({
        "direction": env.observation_space["direction"],
        "image": env.observation_space["image"],
        "mission": gym.spaces.Text(
            max_length=max_len,
            charset=str(alphanumeric) + ' '
        )
    })

    env = DataCollector(env, record_infos=True, observation_space=obs_space, data_format="arrow")

    step_lower_bound = 1_000_000 if random else 10_000
    step_id = 0
    while step_id < step_lower_bound:
        print("STEP", step_id)
        seed = np.random.randint(2**31 - 1)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        policy = RandomPolicy(env) if random else ExpertPolicy(env)
        done = False
        while not done:
            step_id += 1
            action = policy.get_action()
            obs, rew, terminated, truncated, info = env.step(action)
            done = terminated or truncated

    dataset_id = f"minigrid/fourrooms{'-random' if random else ''}-v0"

    dataset = env.create_dataset(
        dataset_id=dataset_id,
        algorithm_name="RandomPolicy" if random else "ExpertPolicy",
        author="Omar G. Younis",
        author_email="omar.younis98@gmail.com",
        code_permalink="https://github.com/rodrigodelazcano/d4rl-minari-dataset-generation",
    )

    print(f"Checking {dataset_id}:")
    run_all_checks(dataset, check_identical=False)


if __name__ == "__main__":
    args = parser.parse_args()
    generate_dataset(random=args.random)
