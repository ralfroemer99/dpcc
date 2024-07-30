import struct
import gymnasium
import glob
import os
import numpy as np
from minari import StepDataCallback, DataCollectorV0
import minari


DIR_PATH = {'complete': '*microwave_kettle_switch_slide', 'partial': '*', 'mixed': '*'}
GOAL_TASKS = {'complete': ['microwave', 'kettle', 'light switch', 'slide cabinet'], 'partial': ['microwave', 'kettle', 'light switch', 'slide cabinet'], 'mixed': ['microwave', 'kettle', 'bottom burner', 'light switch']}


def parse_mjl_logs(read_filename, skipamount):
    print(f'READ FILE {read_filename}')
    with open(read_filename, mode='rb') as file:
        fileContent = file.read()
    headers = struct.unpack('iiiiiii', fileContent[:28])
    nq = headers[0]
    nv = headers[1]
    nu = headers[2]
    nmocap = headers[3]
    nsensordata = headers[4]
    nuserdata = headers[5]
    name_len = headers[6]
    name = struct.unpack(str(name_len) + 's', fileContent[28:28+name_len])[0]
    rem_size = len(fileContent[28 + name_len:])
    num_floats = int(rem_size/4)
    dat = np.asarray(struct.unpack(str(num_floats) + 'f', fileContent[28+name_len:]))
    recsz = 1 + nq + nv + nu + 7*nmocap + nsensordata + nuserdata
    if rem_size % recsz != 0:
        print("ERROR")
    else:
        dat = np.reshape(dat, (int(len(dat)/recsz), recsz))
        dat = dat.T

    time = dat[0,:][::skipamount] - 0*dat[0, 0]
    qpos = dat[1:nq + 1, :].T[::skipamount, :]
    qvel = dat[nq+1:nq+nv+1,:].T[::skipamount, :]
    ctrl = dat[nq+nv+1:nq+nv+nu+1,:].T[::skipamount,:]
    mocap_pos = dat[nq+nv+nu+1:nq+nv+nu+3*nmocap+1,:].T[::skipamount, :]
    mocap_quat = dat[nq+nv+nu+3*nmocap+1:nq+nv+nu+7*nmocap+1,:].T[::skipamount, :]
    sensordata = dat[nq+nv+nu+7*nmocap+1:nq+nv+nu+7*nmocap+nsensordata+1,:].T[::skipamount,:]
    userdata = dat[nq+nv+nu+7*nmocap+nsensordata+1:,:].T[::skipamount,:]

    data = dict(nq=nq,
               nv=nv,
               nu=nu,
               nmocap=nmocap,
               nsensordata=nsensordata,
               name=name,
               time=time,
               qpos=qpos,
               qvel=qvel,
               ctrl=ctrl,
               mocap_pos=mocap_pos,
               mocap_quat=mocap_quat,
               sensordata=sensordata,
               userdata=userdata,
               logName = read_filename
               )
    return data


class KitchenStepDataCallback(StepDataCallback):
    def __call__(self, env, obs, info, action=None, rew=None, terminated=None, truncated=None):
        step_data = super().__call__(env, obs, info, action, rew, terminated, truncated)
        
        # Add the simulation control input to the dataset
        step_data['ctrl'] = env.data.ctrl
        
        return step_data


for dst, dir in DIR_PATH.items():
    print(f'CREATING DATASET: {dst}')
    if dst=='mixed' or dst=='partial':
        max_episode_steps = 450
    else:
        max_episode_steps = 280
    env = gymnasium.make('FrankaKitchen-v1', remove_task_when_completed=False, terminate_on_tasks_completed=False, tasks_to_complete=GOAL_TASKS[dst], max_episode_steps=max_episode_steps)
    env = DataCollectorV0(env, step_data_callback=KitchenStepDataCallback, record_infos=False)
    act_mid = np.zeros(9)
    act_rng = np.ones(9)*2
    dataset = None
    pattern = f'kitchen_demos_multitask/{dir}'
    max_steps_episode = 0
    demo_subdirs = sorted(glob.glob(pattern))
    for subdir in demo_subdirs:
        demo_files = glob.glob(os.path.join(subdir, '*.mjl'))
        
        print(f'Demo: {subdir}')
        for demo_file in demo_files:
            episode_steps = 0
            data = parse_mjl_logs(demo_file, 40)
            obs, _ = env.reset(seed=123)
            if data['ctrl'].shape[0] > max_steps_episode:
                max_steps_episode = data['ctrl'].shape[0]
            for i_frame in range(data['ctrl'].shape[0] - 1):
                # Construct the action
                ctrl = (data['ctrl'][i_frame] - obs['observation'][:9])/(env.robot_env.frame_skip*env.robot_env.model.opt.timestep)
                act = (ctrl - act_mid) / act_rng
                act = np.clip(act, -0.999, 0.999)
                obs, reward, terminated, truncated, env_info = env.step(act)

                episode_steps += 1

    dataset = minari.create_dataset_from_collector_env(collector_env=env, dataset_id=f'kitchen-{dst}-v0', code_permalink="https://github.com/rodrigodelazcano/d4rl-minari-dataset-generation", author="Rodrigo de Lazcano", author_email="rperezvicente@farama.org")
    env.close()      
    print(f'MAX EPISODE STEPS: {max_steps_episode}')
            
