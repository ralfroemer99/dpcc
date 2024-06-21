import os
import collections
import numpy as np
import minari
import pickle
import pdb

from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)

@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

# with suppress_output():
    ## d4rl prints out a variety of warnings
    # import d4rl
 
#-----------------------------------------------------------------------------#
#-------------------------------- general api --------------------------------#
#-----------------------------------------------------------------------------#

# def load_environment(name):
#     if type(name) != str:
#         ## name is already an environment
#         return name
#     with suppress_output():
#         wrapped_env = gym.make(name)
#     env = wrapped_env.unwrapped
#     env.max_episode_steps = wrapped_env._max_episode_steps
#     env.name = name
#     return env

def convert_minari_to_d4rl(dataset_minari):
    episodes_generator = dataset_minari.iterate_episodes()

    dataset = {}
    keys = ['observations', 'actions', 'rewards', 'terminals', 'timeouts']

    dataset = {key: [] for key in keys}

    for episode in episodes_generator:
        # dataset['observations'] = np.concatenate([episode.observations[key] for key in episode.observations], axis=1)
        dataset['observations'] = np.concatenate((episode.observations['observation'], episode.observations['desired_goal']), axis=1)
        dataset['actions'].append(episode.actions)
        dataset['rewards'].append(episode.rewards)
        dataset['terminals'].append(episode.terminations)
        # dataset['timeouts'].append(episode['timeouts'])

    return dataset


def get_dataset(env):
    if type(env) == str:
        dataset_minari = minari.load_dataset(env, download=True)

        dataset = dataset_minari.iterate_episodes()
        # dataset = convert_minari_to_d4rl(dataset_minari)
        # with open('data/' + env + '_dataset.pkl', 'rb') as f:
        #     dataset = pickle.load(f)
    else:
        dataset = env.get_dataset()

    return dataset

def sequence_dataset(env, preprocess_fn):
    """
    Returns an iterator through trajectories.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """

    if type(env) == str:
        dataset = minari.load_dataset(env, download=True)
        episodes_generator = dataset.iterate_episodes()

        for episode in episodes_generator:
            if type(episode.observations) == dict:      # Minari dataset
                # observations = np.concatenate([episode.observations[key] for key in episode.observations], axis=1)
                observations = np.concatenate((episode.observations['observation'], episode.observations['desired_goal']), axis=1)
                goal_dim = episode.observations['desired_goal'].shape[1]
                observations[0, -goal_dim:] = observations[1, -goal_dim:]      # Ensure that the goal is already set in the first timestep (from previous episode)
            else:
                observations = episode.observations

            if observations.shape[0] == episode.actions.shape[0] + 1:
                observations = observations[:-1]
            
            episode_data = {
                'observations': observations,
                'actions': episode.actions,
                'rewards': episode.rewards,
                'terminals': episode.terminations
            }
            yield episode_data
    else:
        raise NotImplementedError

    # dataset = get_dataset(env)
    # dataset = preprocess_fn(dataset)

    # # N = dataset['rewards'].shape[0]
    # N = len(dataset['rewards'])
    # data_ = collections.defaultdict(list)

    # # The newer version of the dataset adds an explicit
    # # timeouts field. Keep old method for backwards compatability.
    # use_timeouts = 'timeouts' in dataset

    # episode_step = 0
    # for i in range(N):
    #     done_bool = bool(dataset['terminals'][i])
    #     if use_timeouts:
    #         final_timestep = dataset['timeouts'][i]
    #     else:
    #         final_timestep = (episode_step == env._max_episode_steps - 1)

    #     for k in dataset:
    #         if 'metadata' in k: continue
    #         data_[k].append(dataset[k][i])

    #     if done_bool or final_timestep:
    #         episode_step = 0
    #         episode_data = {}
    #         for k in data_:
    #             episode_data[k] = np.array(data_[k])
    #         yield episode_data
    #         data_ = collections.defaultdict(list)

    #     episode_step += 1
