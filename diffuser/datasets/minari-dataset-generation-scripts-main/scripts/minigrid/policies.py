import numpy as np

class RandomPolicy:

    def __init__(self, env) -> None:
        self.env = env

    def get_action(self):
        return self.env.action_space.sample()



class ExpertPolicy:

    def __init__(self, env) -> None:
        self.env = env.unwrapped
        grid_obj = self.env.grid
        goal_grid = [cell is not None and cell.type == "goal" for cell in grid_obj.grid]
        goal_grid = np.array(goal_grid)
        goal_grid = goal_grid.reshape(grid_obj.height, grid_obj.width)
        wall_grid = [cell is not None and cell.type == "wall" for cell in grid_obj.grid]
        wall_grid = np.array(wall_grid)
        wall_grid = wall_grid.reshape(grid_obj.height, grid_obj.width)
        self.values = self._value_iteration(goal_grid, wall_grid)

    def _value_iteration(self, goal_grid, wall_grid):
        num_rows, num_cols = goal_grid.shape
        optimal_values = np.full((num_rows, num_cols), -float('inf'))
        visited = np.zeros((num_rows, num_cols), dtype=np.bool_)
        goal_pos = np.argwhere(goal_grid)
        assert len(goal_pos) == 1
        goal_y, goal_x = goal_pos[0]
        optimal_values[goal_y, goal_x] = 1
        queue = [(goal_y, goal_x)]

        while len(queue) > 0:
            node = queue.pop(0)
            visited[node] = True
            if node[1] == self.env.agent_pos[0] and node[0] == self.env.agent_pos[1]:
                return optimal_values
            for move in {(0, -1), (-1, 0), (0, 1), (1, 0)}:
                new_node = node[0] + move[0], node[1] + move[1]
                if new_node[0] >= num_rows or new_node[1] >= num_cols:
                    continue
                if not wall_grid[new_node] and not visited[new_node]:
                    optimal_values[new_node] = optimal_values[node] - 1
                    queue.append(new_node)
                

        return optimal_values

    def get_action(self):
        x, y = self.env.agent_pos
        best_move = (0, -1)
        for move in {(-1, 0), (0, 1), (1, 0)}:
            if self.values[y + move[0], x + move[1]] > self.values[y + best_move[0], x + best_move[1]]:
                best_move = move
        
        if best_move[1] == self.env.dir_vec[0] and best_move[0] == self.env.dir_vec[1]:
            return 2
        if best_move[1] == self.env.dir_vec[1] and best_move[0] == -self.env.dir_vec[0]:
            return 0
        else:
            return 1