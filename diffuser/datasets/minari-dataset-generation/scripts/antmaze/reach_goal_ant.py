import gymnasium as gym
from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames
from gymnasium import spaces

import numpy as np
import tempfile
import xml.etree.ElementTree as ET

import sys
from gymnasium.envs.mujoco.ant_v4 import AntEnv
from gymnasium.utils.ezpickle import EzPickle

from typing import Dict, List, Optional, Union
from os import path


class GoalReachAnt(gym.Env, EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        continuing_task: bool = True,
        goal_region_radius: float = 10.,
        goal_threshold: float = 0.30,
        forward_reward_scale: float = 10.,
        **kwargs,
    ):
        self.goal_region_radius = goal_region_radius
        self.goal_threshold = goal_threshold
        self.continuing_task = continuing_task
        self.forward_reward_scale = forward_reward_scale
        
        # Get the ant.xml path from the Gymnasium package
        ant_xml_file_path = path.join(
            path.dirname(sys.modules[AntEnv.__module__].__file__), "assets/ant.xml"
        )
        
        tree = ET.parse(ant_xml_file_path)
        worldbody = tree.find(".//worldbody")
        
        # Add target site for visualization
        ET.SubElement(
            worldbody,
            "site",
            name="target",
            pos=f"0 0 0.75",
            size="0.25",
            rgba="1 0 0 0.7",
            type="sphere",
        )
        
        # Save new xml with maze to a temporary file
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_xml_path = path.join(path.dirname(tmp_dir), "ant_maze.xml")
            tree.write(temp_xml_path)
        
        self.ant_env = AntEnv(
            xml_file=temp_xml_path,
            exclude_current_positions_from_observation=False,
            render_mode=render_mode,
            reset_noise_scale=0.15,
            **kwargs,
        )
        
        self._model_names = MujocoModelNames(self.ant_env.model)
        self.target_site_id = self._model_names.site_name2id["target"]
        self.action_space = self.ant_env.action_space
        obs_shape: tuple = self.ant_env.observation_space.shape
        self.observation_space =spaces.Box(
                    -np.inf, np.inf, shape=(obs_shape[0],), dtype="float64"
                )

        self.render_mode = render_mode

        EzPickle.__init__(
            self,
            render_mode,
            continuing_task,
            **kwargs,
        )
    
    def reset(self, *, seed: Optional[int] = None, **kwargs):
        super().reset(seed=seed)
        self.goal = self._generate_goal()
        self.ant_env.model.site_pos[self.target_site_id] = np.append(self.goal, 0.75)
        obs, info = self.ant_env.reset(seed=seed)
        obs_dict, _ = self._get_obs(obs)

        return obs_dict, info
    
    def step(self, action):
        ant_obs, _, _, _, info = self.ant_env.step(action)
        obs, achieved_goal = self._get_obs(ant_obs)

        terminated = self.compute_terminated(achieved_goal, self.goal, info)
        truncated = self.compute_truncated(achieved_goal, self.goal, info)
        reward = self.compute_reward(achieved_goal, self.goal, info)
        
        # Create a new goal, if necessary, and update the observation
        if self.continuing_task and self._within_goal_threshold(achieved_goal, self.goal):
            self.goal = self._generate_goal()
            self.ant_env.model.site_pos[self.target_site_id] = np.append(self.goal, 0.75)
            obs, _  = self._get_obs(ant_obs)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _get_obs(self, ant_obs: np.ndarray):
        achieved_goal = ant_obs[:2]
        goal_direction = self.goal - achieved_goal
        observation = np.concatenate([ant_obs[2:], goal_direction])
        return observation, achieved_goal

    def _generate_goal(self):
        th = 2 * np.pi * self.np_random.uniform()
        radius = self.goal_region_radius * self.np_random.uniform()
        return radius * np.array([np.cos(th), np.sin(th)])

    def render(self):
        return self.ant_env.render()

    def close(self):
        super().close()
        self.ant_env.close()
    
    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info
    ) -> float:
        """Compute reward based on velocity in direction of goal.
        
           Reward is based on the original Ant env rewards, but replacing the original
           forward_reward with the velocity in the goal direction. This velocity is scaled
           by a factor of 10 (by default) as the average velocity is slower than in the
           original env due to the need to turn.
        """
        goal_vector = desired_goal - achieved_goal
        velocity_vector = [info["x_velocity"], info["y_velocity"]]
        reward_forward = np.dot(velocity_vector, goal_vector)/np.linalg.norm(goal_vector)
        return self.forward_reward_scale*reward_forward + info["reward_ctrl"] + info["reward_survive"]
    
    def compute_truncated(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info
    ) -> bool:
        return False

    def compute_terminated(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info
    ) -> bool:
        if self.continuing_task:
            return False
        return self._within_goal_threshold(achieved_goal, desired_goal, info)
    
    
    def _within_goal_threshold(self, achieved_goal, desired_goal):
        return bool(np.linalg.norm(achieved_goal - desired_goal) <= self.goal_threshold)