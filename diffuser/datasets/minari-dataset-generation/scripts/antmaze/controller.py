import os
import sys
import numpy as np

# Import maze_solver from pointmaze
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pointmaze.maze_solver import DFS, QIteration



class WaypointController:
    """Generic agent controller to follow waypoints in the maze.

    Inspired by https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/pointmaze/waypoint_controller.py
    """

    def __init__(
        self, maze, model_callback, maze_solver="DFS", waypoint_threshold=0.45
    ):
        self.global_target_xy = np.empty(2)
        self.maze = maze
        if maze_solver == "DFS":
            self.maze_solver = DFS(maze=self.maze)
        else:
            self.maze_solver = QIteration(maze=self.maze)

        self.model_callback = model_callback
        self.waypoint_threshold = waypoint_threshold
        self.waypoint_targets = None

    def compute_action(self, obs):
        # Check if we need to generate new waypoint path due to change in global target
        if (
            np.linalg.norm(self.global_target_xy - obs["desired_goal"]) > 1e-3
            or self.waypoint_targets is None
        ):
            # Convert xy to cell id
            achived_goal_cell = tuple(self.maze.cell_xy_to_rowcol(obs["achieved_goal"]))
            self.global_target_id = tuple(
                self.maze.cell_xy_to_rowcol(obs["desired_goal"])
            )
            
            self.global_target_xy = obs["desired_goal"]

            self.waypoint_targets = self.maze_solver.generate_path(
                achived_goal_cell, self.global_target_id
            )
            # Check if the waypoint dictionary is empty
            # If empty then the ball is already in the target cell location
            if self.waypoint_targets:
                self.current_control_target_id = self.waypoint_targets[
                    achived_goal_cell
                ]
                # If target is global goal go directly to goal position
                if self.current_control_target_id == self.global_target_id:
                    self.current_control_target_xy = obs['desired_goal']
                else:
                    self.current_control_target_xy = self.maze.cell_rowcol_to_xy(
                        np.array(self.current_control_target_id)
                    ) - np.random.uniform(size=(2,)) * 0.1
            else:
                self.waypoint_targets[
                    self.current_control_target_id
                ] = self.current_control_target_id
                self.current_control_target_id = self.global_target_id
                self.current_control_target_xy = self.global_target_xy

        # Check if we need to go to the next waypoint
        dist = np.linalg.norm(self.current_control_target_xy - obs["achieved_goal"])
        if (
            dist <= self.waypoint_threshold
            and self.current_control_target_id != self.global_target_id
        ):
            self.current_control_target_id = self.waypoint_targets[
                self.current_control_target_id
            ]
            # If target is global goal go directly to goal position
            if self.current_control_target_id == self.global_target_id:

                self.current_control_target_xy = obs['desired_goal']
            else:
                self.current_control_target_xy = (
                    self.maze.cell_rowcol_to_xy(
                        np.array(self.current_control_target_id)
                    )
                    - np.random.uniform(size=(2,)) * 0.1
                )  
        action = self.model_callback(obs, self.current_control_target_xy)
        return action
