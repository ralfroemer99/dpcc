import h5py
import minari
import os
import mujoco
import numpy as np
import gymnasium as gym
from scripts.kitchen.utils import AdroitStepDataCallback, download_dataset_from_url
from minari import DataCollector
    

if __name__ == "__main__":
    # create directory to store the original d4rl datasets
    if not os.path.exists('d4rl_datasets'):
        os.makedirs('d4rl_datasets')
    
    # human episode steps vary between 250-300, for expert all trajectories have lenght of 200, and for cloned
    max_episode_steps = {'human': 530, 'cloned': 530, 'expert': 200}
    
    for dset in ['human', 'cloned', 'expert']:
        d4rl_dataset_name = 'relocate-' + dset + '-v1'
        minari_dataset_name = 'relocate/' + dset + '-v0'
        
        d4rl_url = f'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg_v1/{d4rl_dataset_name}.hdf5'
        download_dataset_from_url(d4rl_url)   
        env = gym.make('AdroitHandRelocate-v1', max_episode_steps=650)

        print(f'Recreating {d4rl_dataset_name} D4RL dataset to Minari {minari_dataset_name}')
        with h5py.File(f'd4rl_datasets/{d4rl_dataset_name}.hdf5', 'r') as f:
            actions = f['actions'][:]
            observations = f['observations'][:]
            timeouts = f['timeouts'][:]
            qposes = f['infos']['qpos'][:]
            qvels = f['infos']['qvel'][:]
        
        # we need to recreate the target and obj intial state from the observations, since in the d4rl dataset
        # these values are the same for all episodes (https://github.com/Farama-Foundation/D4RL/issues/196)
        palm_poses = np.zeros((qposes.shape[0],3))
        obj_poses = np.zeros((qposes.shape[0], 3))
        target_poses = np.zeros((qposes.shape[0], 3))

        reset_called = True
        last_episode_step = 0
        env.reset()
        for i, (timeout, action, qpos, qvel, observation) in enumerate(zip(timeouts, actions, qposes, qvels, observations)):
            env.set_state(qpos, qvel)
            mujoco.mj_step(env.model, env.data, nstep=env.frame_skip)
            palm_pos = env.data.site_xpos[env.S_grasp_site_id].ravel().copy()
            obj_poses[i, :] = palm_pos-observation[30:33]
            target_poses[i, :] = obj_poses[i, :]-observation[-3:]

        env.close()

        env = gym.make('AdroitHandRelocate-v1', max_episode_steps=max_episode_steps[dset])
        env = DataCollector(env, step_data_callback=AdroitStepDataCallback, record_infos=True, max_buffer_steps=10000)

        reset_called = True
        last_episode_step = 0
        for i, (timeout, observation, action, target_pos, obj_pos, qpos, qvel) in enumerate(zip(timeouts, observations, actions, target_poses, obj_poses, qposes, qvels)):
            if reset_called:
                state_dict = {'qpos': qpos, 'qvel': qvel, 'obj_pos': obj_pos, 'target_pos': target_pos}
                env.reset(options={'initial_state_dict': state_dict})
                reset_called=False
            # assert not np.allclose(observation, obs, rtol=1e-2, atol=1e-4)
            obs, rew, terminated, truncated, info = env.step(action)
            
            if i % 50000 == 0:
                print(i)
                
            if timeout:
                reset_called = True

        env.create_dataset(
            dataset_name=minari_dataset_name,
            code_permalink="https://github.com/rodrigodelazcano/d4rl-minari-dataset-generation",
            author="Rodrigo de Lazcano",
            author_email="rperezvicente@farama.org"
        )
        
        env.close()

    minari.list_local_datasets()