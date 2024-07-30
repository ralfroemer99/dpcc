import os
import h5py
import minari
import gymnasium as gym
from scripts.kitchen.utils import AdroitStepDataCallback, download_dataset_from_url
from minari import DataCollector


if __name__ == "__main__":
    # create directory to store the original d4rl datasets
    if not os.path.exists('d4rl_datasets'):
        os.makedirs('d4rl_datasets')
    
    # human episode steps vary between 250-300, for expert all trajectories have lenght of 200, and for cloned 250-300
    max_episode_steps = {'human': 300, 'cloned': 300, 'expert': 200}
    
    for dset in ['human', 'cloned', 'expert']:
        d4rl_dataset_name = 'door-' + dset + '-v1'
        minari_dataset_name = 'door/' + dset + '-v0'
        
        d4rl_url = f'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg_v1/{d4rl_dataset_name}.hdf5'
        download_dataset_from_url(d4rl_url)

        env = gym.make('AdroitHandDoor-v1', max_episode_steps=max_episode_steps[dset])
        env = DataCollector(env, step_data_callback=AdroitStepDataCallback, record_infos=True, max_buffer_steps=200000)
        
        print(f'Recreating {d4rl_dataset_name} D4RL dataset to Minari {minari_dataset_name}')
        with h5py.File(f'd4rl_datasets/{d4rl_dataset_name}.hdf5', 'r') as f:
            qposes = f['infos']['qpos'][:]
            qvels = f['infos']['qvel'][:]
            actions = f['actions'][:]
            observations = f['observations'][:]
            door_poses = f['infos']['door_body_pos'][:]
            timeouts = f['timeouts'][:]
            
        reset = True
        for i, (timeout, observation, action, door_pos, qpos, qvel) in enumerate(zip(timeouts, observations, actions, door_poses, qposes, qvels)):
            if reset:
                state_dict = {'qpos': qpos, 'qvel': qvel, 'door_body_pos': door_pos}
                env.reset(options={'initial_state_dict': state_dict})      
                reset=False
            # assert np.allclose(observation, obs, rtol=1e-2, atol=1e-4)

            if i % 50000 == 0:
                print(i)
                
            obs, rew, terminated, truncated, info = env.step(action)
                
            if timeout:
                reset = True

        env.create_dataset(
            dataset_name=minari_dataset_name,
            code_permalink="https://github.com/rodrigodelazcano/d4rl-minari-dataset-generation",
            author="Rodrigo de Lazcano",
            author_email="rperezvicente@farama.org"
        )    

        env.close()

    minari.list_local_datasets()