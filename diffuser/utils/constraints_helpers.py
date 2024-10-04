import numpy as np

def formulaze_halfspace_constraints(constraint, enlarge_constraints, trajectory_dim, action_dim, obs_indices):
    m = (constraint[1][1] - constraint[0][1]) / (constraint[1][0] - constraint[0][0])
    n = [-1, 1/m] / np.linalg.norm([-1, 1/m])
    points_enlarged = [constraint[0] + enlarge_constraints * n, constraint[1] + enlarge_constraints * n]
    d = points_enlarged[0][1] - m * points_enlarged[0][0]
    # d = constraint[0][1] - m * constraint[0][0]
    C_row = np.zeros(trajectory_dim)
    if constraint[2] == 'below':
        C_row[obs_indices['x'] + action_dim] = -m
        C_row[obs_indices['y'] + action_dim] = 1
    elif constraint[2] == 'above':
        C_row[obs_indices['x'] + action_dim] = m
        C_row[obs_indices['y'] + action_dim] = -1
        d *= -1
    return C_row, d