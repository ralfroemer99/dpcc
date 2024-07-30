"""
This script runs standard sanity checks that apply to all Minari datasets.

Usage:

python check_dataset.py <dataset_id>
"""

import argparse

import minari
import numpy as np
import scipy
from gymnasium import spaces
from minigrid.core.mission import MissionSpace


def get_infos(dataset, key):
    """Get a list of the infos[key] for each episode."""
    return list(dataset._data.apply(lambda ep: ep["infos"][key][()]))


def print_avg_returns(dataset):
    """Print returns for manual sanity checking."""
    all_returns = []
    num_truncations = 0
    num_terminations = 0

    for ep in dataset:
        all_returns.append(np.sum(ep.rewards))
        num_truncations += np.sum(ep.truncations)
        num_terminations += np.sum(ep.terminations)

    print("  | Average returns:", np.mean(all_returns))
    print("  | Num truncations:", num_truncations)
    print("  | Num terminations:", num_terminations)


def _check_identical(i, key, values, ignore_keys):
    if isinstance(values, dict):
        for k, v in values.items():
            if k not in ignore_keys:
                _check_identical(i, f"{key}['{k}']", v, ignore_keys)
    else:
        assert isinstance(values, np.ndarray)

        if len(values) < 3:
            return

        values_0 = values[0]
        values_mid = values[values.shape[0] // 2]
        values_last = values[-1]

        test_values = np.c_[values_0, values_mid, values_last].T
        dists = scipy.spatial.distance.pdist(test_values)
        not_same = dists > 0

        message = (
            f"Some of first/mid/last values of ep[{i}].{key} "
            f"(length {len(values)}) are identical:\n{test_values}"
        )
        assert np.all(not_same), message


def check_identical_values(
    dataset, check_keys=["actions", "observations"], ignore_keys=["desired_goal"]
):
    """Check that values of actions/observations in episodes are not identical."""
    for i, episode in enumerate(dataset):
        for key in check_keys:
            _check_identical(i, key, getattr(episode, key), ignore_keys)


def _check_shape(i, key, value, inner_space, outer_dim_size):
    # If it's a Dict or Tuple, recurse and check all values have the correct shape
    if isinstance(inner_space, spaces.Dict):
        assert isinstance(value, dict)
        for k, v in value.items():
            _check_shape(i, f"{key}['{k}']", v, inner_space[k], outer_dim_size)
    elif isinstance(inner_space, spaces.Tuple):
        assert isinstance(value, tuple)
        for i, x in enumerate(value):
            _check_shape(i, f"{key}[{i}]", x, inner_space[i], outer_dim_size)
    elif isinstance(inner_space, MissionSpace):
        # MissionSpaces do not have a shape -- check they are one-dimensional lists
        assert isinstance(value, list)
        _check_shape(i, key, np.array(value), None, outer_dim_size)
    else:
        assert isinstance(value, np.ndarray), f"episode[{i}].{key} is not np.ndarray"

        if inner_space is None or isinstance(inner_space, spaces.Discrete):
            expected_shape = (outer_dim_size,)
        else:
            assert inner_space.shape is not None
            expected_shape = (outer_dim_size, *inner_space.shape)

        shape_message = (
            f"Expected episode[{i}].{key} to have shape "
            f"{expected_shape}, got {value.shape}"
        )
        assert value.shape == expected_shape, shape_message


def check_episode_shapes(dataset):
    """Check that all entries in each episode have correct shape.
    From https://minari.farama.org/content/dataset_standards/,
    each entry in an episode is of type np.ndarray, with shape:
      - actions:      (num_steps, action_space_shape)
      - observations: (num_steps + 1, obs_space_shape)
      - rewards:      (num_steps)
      - terminations: (num_steps)
      - truncations:  (num_steps)
    The actions and observations are possibly nested e.g. in a dict, so we check
    that the innermost shapes match.
    """
    # Check episodes
    for i, ep in enumerate(dataset):
        num_steps = ep.total_steps
        env = dataset.recover_environment()

        # Check each key has the correct shape
        target_shapes = {
            "actions": (num_steps, env.action_space),
            "observations": (num_steps + 1, env.observation_space),
            "rewards": (num_steps, None),
            "terminations": (num_steps, None),
            "truncations": (num_steps, None),
        }

        for key, (outer_dim, inner_shape) in target_shapes.items():
            _check_shape(i, key, getattr(ep, key), inner_shape, outer_dim)


def check_infos_consistency(dataset):
    """Check that all infos dicts have consistent keys and shapes."""
    first_ep = next(iter(dataset))

    if not hasattr(first_ep, "infos"):
        return

    info_shapes = {k: v.shape[1:] for k, v in first_ep.infos.items()}

    for i, ep in enumerate(dataset):
        num_steps = ep.total_steps
        infos = ep.infos

        assert (
            infos.keys() == info_shapes.keys()
        ), f"episode[{i}] and episode[0] don't have the same info keys"

        for k, v in infos.items():
            expected_shape = (num_steps + 1, *info_shapes[k])
            assert (
                v.shape == expected_shape
            ), f"episode[{i}].infos['{k}'] has shape {v.shape}, expected {expected_shape}"


def run_all_checks(dataset, verbose=True, check_identical=True):
    """Run all of the standard Minari checks and print results."""
    passed = True

    to_check = check_fns if check_identical else check_fns_no_identical

    for check_fn in to_check:
        try:
            check_fn(dataset)
        except AssertionError as err:
            passed = False
            print("FAILED:", check_fn.__name__)
            print(err)
        else:
            if verbose:
                print(f"PASSED: {check_fn.__name__}")

    return passed


check_fns_no_identical = [
    print_avg_returns,
    check_episode_shapes,
    check_infos_consistency,
]
check_fns = check_fns_no_identical + [check_identical_values]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_name", type=str, help="name of minari dataset to check"
    )
    parser.add_argument(
        "--ignore-identical",
        action="store_true",
        help="don't check for identical dataset values",
    )
    args = parser.parse_args()

    dataset = minari.load_dataset(args.dataset_name)

    print("\nChecking:", args.dataset_name)
    passed = run_all_checks(dataset, check_identical=not args.ignore_identical)

    print("All tests passed" if passed else "Tests FAILED")
