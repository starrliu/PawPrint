"""Utility functions for trajectory analysis."""

import numpy as np

def calculate_movement(x: list, y: list) -> float:
    """Calculate the total movement distance from a sequence of x, y coordinates.

    Args:
        x (list): List of x coordinates.
        y (list): List of y coordinates.

    Returns:
        float: Total distance moved. Returns 0 if there are insufficient valid points.

    Note:
        - NaN values in the trajectory are handled by skipping those segments
        - At least two valid consecutive points are needed to calculate movement
    """
    if len(x) != len(y):
        raise ValueError("x and y coordinates must have the same length")
    
    total_movement = 0
    
    for i in range(len(x) - 1):
        # Check if we have valid coordinates for this segment
        if not any(np.isnan([x[i], x[i+1], y[i], y[i+1]])):
            # Calculate Euclidean distance between consecutive points
            segment_dist = np.sqrt((x[i+1] - x[i])**2 + (y[i+1] - y[i])**2)
            total_movement += segment_dist
            
    return total_movement 