#!/usr/bin/env python3
"""Script for visualizing distance matrix between mouse pairs."""

import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from pawprint.event.approach import ApproachDetector
from pawprint.data import TrajectoryCollection
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_distance_matrix(trajectory_collection):
    """
    Create distance matrix between all mouse pairs.
    
    Args:
        trajectory_collection: TrajectoryCollection object
    
    Returns:
        tuple: (distance_matrix, identities)
    """
    identities = trajectory_collection.identities
    n_mice = len(identities)
    distances = np.zeros((n_mice, n_mice))
    
    print(f"Calculating distance matrix for {n_mice} mice...")
    
    for i, identity_a in enumerate(identities):
        for j, identity_b in enumerate(identities):
            if i != j:  # Skip self-distances
                distance = np.array(trajectory_collection.to_distance(identity_a, identity_b))
                # Filter out nan values
                distance = distance[~np.isnan(distance)]
                if len(distance) > 0:
                    distances[i, j] = np.percentile(distance, 50)  # Median distance
                else:
                    distances[i, j] = np.nan
            else:
                distances[i, j] = 0  # Self-distance is 0
    
    return distances, identities

def plot_distance_matrix(distance_matrix, identities, save_path=None):
    """
    Plot distance matrix as a heatmap with improved styling.
    
    Args:
        distance_matrix: 2D array of distances
        identities: List of mouse identities
        save_path: Path to save the plot (optional)
    """
    n_mice = len(identities)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap with custom colormap
    im = ax.imshow(distance_matrix, cmap='YlOrRd', aspect='auto', vmin=0)
    
    # Customize the plot
    ax.set_title('Mouse Pair Distance Matrix\n(Median Distance in cm)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Mouse ID', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mouse ID', fontsize=14, fontweight='bold')
    
    # Set tick labels
    ax.set_xticks(range(n_mice))
    ax.set_yticks(range(n_mice))
    ax.set_xticklabels([f'Mouse {id}' for id in identities])
    ax.set_yticklabels([f'Mouse {id}' for id in identities])
    
    # Rotate x-axis labels
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='both', labelsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Distance (cm)', fontsize=12, fontweight='bold')
    
    # Add text annotations for non-zero distances
    for i in range(n_mice):
        for j in range(n_mice):
            if i != j and not np.isnan(distance_matrix[i, j]):
                text = ax.text(j, i, f'{distance_matrix[i, j]:.1f}',
                              ha="center", va="center", color="white", 
                              fontsize=10, fontweight='bold')
            elif i == j:
                text = ax.text(j, i, '0.0',
                              ha="center", va="center", color="white", 
                              fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distance matrix plot saved to {save_path}")
    
    plt.show()

def print_distance_statistics(distance_matrix, identities):
    """
    Print summary statistics for the distance matrix.
    
    Args:
        distance_matrix: 2D array of distances
        identities: List of mouse identities
    """
    print("\n" + "="*60)
    print("DISTANCE MATRIX STATISTICS")
    print("="*60)
    
    # Calculate statistics excluding diagonal and NaN values
    valid_distances = distance_matrix[distance_matrix > 0]
    valid_distances = valid_distances[~np.isnan(valid_distances)]
    
    print(f"Number of mouse pairs: {len(valid_distances)}")
    print(f"Average distance: {np.mean(valid_distances):.2f} ± {np.std(valid_distances):.2f} cm")
    print(f"Median distance: {np.median(valid_distances):.2f} cm")
    print(f"Min distance: {np.min(valid_distances):.2f} cm")
    print(f"Max distance: {np.max(valid_distances):.2f} cm")
    
    # Find closest and farthest pairs
    min_idx = np.unravel_index(np.nanargmin(distance_matrix + np.eye(len(distance_matrix)) * np.inf), distance_matrix.shape)
    max_idx = np.unravel_index(np.nanargmax(distance_matrix), distance_matrix.shape)
    
    print(f"\nClosest pair: Mouse {identities[min_idx[0]]} - Mouse {identities[min_idx[1]]} ({distance_matrix[min_idx]:.2f} cm)")
    print(f"Farthest pair: Mouse {identities[max_idx[0]]} - Mouse {identities[max_idx[1]]} ({distance_matrix[max_idx]:.2f} cm)")

def main():
    """Main function to run the distance matrix analysis."""
    
    # Load trajectory data
    print("Loading trajectory data...")
    trajectory_collection = TrajectoryCollection("data/gt_mtrack.csv", fps=30, scale=0.259)
    print(f"Loaded {len(trajectory_collection)} mice trajectories")
    
    # Create distance matrix
    distance_matrix, identities = create_distance_matrix(trajectory_collection)
    
    # Plot distance matrix
    print("\nCreating distance matrix visualization...")
    plot_distance_matrix(
        distance_matrix, 
        identities,
        save_path="distance_matrix.png"
    )
    
    # Print statistics
    print_distance_statistics(distance_matrix, identities)
    
    # Save matrix data
    import pandas as pd
    df_matrix = pd.DataFrame(
        distance_matrix, 
        index=[f'Mouse {id}' for id in identities],
        columns=[f'Mouse {id}' for id in identities]
    )
    df_matrix.to_csv('distance_matrix_data.csv')
    print(f"\nMatrix data saved to distance_matrix_data.csv")

if __name__ == "__main__":
    main()



