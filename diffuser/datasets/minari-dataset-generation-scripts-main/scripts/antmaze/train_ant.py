"""
This script trains the SB3 model given by GoalReachAnt_model.zip

Usage:

python train_ant.py

See --help for full list of options.
"""

import argparse
import glob
import os

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch.nn as nn
import wandb
from gymnasium.envs.registration import register
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from wandb.integration.sb3 import WandbCallback

project_name = "GoalAnt"
checkpoint_dir = "./logs/GoalAnt"
checkpoint_filename = f"{project_name}_model"

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 30_000_000,
    "env_id": "GoalReachAnt-v0",
    "goal_threshold": 0.2,
    "forward_reward_scale": 10.0,
    "n_envs": 16,
    "max_episode_steps": 1000,
    "checkpoint_dir": checkpoint_dir,
    "checkpoint_filename": checkpoint_filename,
    "model_kwargs": {
        "seed": 42,
        "verbose": 1,
        "batch_size": 256,
        "buffer_size": 1000000,
        "ent_coef": "auto_0.1",
        "gamma": 0.991,
        "gradient_steps": 1,
        "learning_rate": 0.0003,
        "learning_starts": 10000,
        "policy_kwargs": {"net_arch": [256] * 4, "activation_fn": nn.ReLU},
        "tau": 0.03,
        "train_freq": 1,
        "use_sde": False,
        "action_noise": NormalActionNoise(mean=[0.0], sigma=[0.3]),
    },
}


def latest_glob(directory, glob_str):
    """Find latest file in directory that matches glob."""
    filenames = glob.glob(f"{directory}/{glob_str}")
    return max(filenames, key=os.path.getctime)


def make_env(run_name:str):
    """Wrapper to create the appropriate environment."""
    env = gym.make(
        config["env_id"],
        render_mode="rgb_array",
        forward_reward_scale=config["forward_reward_scale"],
        goal_threshold=config["goal_threshold"],
    )
    env = RecordVideo(env, f"videos/{run_name}")
    env = Monitor(env)
    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume", type=str, default="", help="WandB id to resume training"
    )
    args = parser.parse_args()

    # Register GoalReachAnt environment
    register(
        id=config["env_id"],
        entry_point="reach_goal_ant:GoalReachAnt",
        max_episode_steps=config["max_episode_steps"],
    )

    run_name = "ant-goal-reach"

    # Set up WandB run
    run = wandb.init(
        project=project_name,
        name=run_name,
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        resume="must" if args.resume != "" else False,
        id=args.resume if args.resume != "" else None,
    )
    
    env = make_vec_env(make_env, n_envs=config["n_envs"], env_kwargs={"run_name": run_name})

    # Support resuming prematurely stopped runs with --resume <wandb run id>
    if wandb.run.resumed:
        # Load locally saved model and replay buffer checkpoint
        checkpoint_load_filepath = latest_glob(
            checkpoint_dir, f"{checkpoint_filename}_*_steps.zip"
        )

        buffer_load_filepath = latest_glob(
            checkpoint_dir, f"{checkpoint_filename}_replay_buffer_*_steps.pkl"
        )

        print(f"Resuming run: {checkpoint_load_filepath}")
        print(f"      Buffer: {buffer_load_filepath}")

        model = SAC.load(
            checkpoint_load_filepath, tensorboard_log=f"runs/{run.id}", env=env
        )
        model.load_replay_buffer(buffer_load_filepath)

        print(f"Replay buffer has has {model.replay_buffer.size()} transitions")
    else:
        model = SAC(
            config["policy_type"],
            env,
            tensorboard_log=f"runs/{run.id}",
            **config["model_kwargs"],
        )

    wandb_callback = WandbCallback(
        gradient_save_freq=100, model_save_path=f"models/{run.id}", verbose=1
    )

    # Save a checkpoint every 1M steps
    checkpoint_callback = CheckpointCallback(
        save_freq=1_000_000 // config["n_envs"],
        save_path=checkpoint_dir,
        name_prefix=checkpoint_filename,
        save_replay_buffer=True,
    )

    # Start training
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=[wandb_callback, checkpoint_callback],
        reset_num_timesteps=False,
    )

    run.finish()
