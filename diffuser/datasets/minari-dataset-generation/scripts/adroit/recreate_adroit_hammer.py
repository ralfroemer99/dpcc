import h5py
import minari
import os
import gymnasium as gym
from scripts.kitchen.utils import AdroitStepDataCallback, download_dataset_from_url
from minari import DataCollector

if __name__ == "__main__":
     # create directory to store the original d4rl datasets
    if not os.path.exists('d4rl_datasets'):
        os.makedirs('d4rl_datasets')
    
    # human episode steps vary between 300-650, for expert all trajectories have lenght of 200, and for cloned
    max_episode_steps = {'human': 625, 'cloned': 625, 'expert': 200}
    
    for dset in ['human', 'expert', 'cloned']:
        d4rl_dataset_name = 'hammer-' + dset + '-v1'
        minari_dataset_name = 'hammer/' + dset + '-v0'
        
        d4rl_url = f'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg_v1/{d4rl_dataset_name}.hdf5'
        download_dataset_from_url(d4rl_url)
        env = gym.make('AdroitHandHammer-v1', max_episode_steps=max_episode_steps[dset])
        env = DataCollector(env, step_data_callback=AdroitStepDataCallback, record_infos=True, max_buffer_steps=200000)

        print(f'Recreating {d4rl_dataset_name} D4RL dataset to Minari {minari_dataset_name}')
        with h5py.File(f'd4rl_datasets/{d4rl_dataset_name}.hdf5', 'r') as f:
            actions = f['actions'][:]
            qposes = f['infos']['qpos'][:]
            qvels = f['infos']['qvel'][:]
            observations = f['observations'][:]
            board_poses = f['infos']['board_pos'][:]
            timeouts = f['timeouts'][:]
        
        reset_called = True
        for i, (timeout, observation, action, board_pos, qpos, qvel) in enumerate(zip(timeouts, observations, actions, board_poses, qposes, qvels)):
            if reset_called:
                state_dict = {'qpos': qpos, 'qvel': qvel, 'board_pos': board_pos}
                env.reset(options={'initial_state_dict': state_dict})
                reset_called=False
            # assert np.allclose(observation, obs, rtol=1e-2, atol=1e-4)

            if i % 50000 == 0:
                print(i)
                
            obs, rew, terminated, truncated, info = env.step(action)
            
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