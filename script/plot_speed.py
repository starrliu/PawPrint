#!/usr/bin/env python3
"""Test script for plotting mouse speed distributions using box plots."""

import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pawprint.data import TrajectoryCollection

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_speed_distributions(trajectory_collection, window_size=10, mode="single", save_path=None):
    """
    Plot speed distributions for each mouse using box plots.
    
    Args:
        trajectory_collection: TrajectoryCollection object
        window_size: Window size for speed calculation
        mode: Speed calculation mode ("linear", "mean", "single")
        save_path: Path to save the plot (optional)
    """
    
    # Calculate speeds for all mice
    speeds_dict = trajectory_collection.to_speed(window_size=window_size, mode=mode)
    
    # Prepare data for plotting
    plot_data = []
    mouse_ids = []
    
    for mouse_id, speeds in speeds_dict.items():
        if speeds:  # Only include mice with valid speed data
            plot_data.extend(speeds)
            mouse_ids.extend([f"Mouse {mouse_id}"] * len(speeds))
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        'Speed (cm/s)': plot_data,
        'Mouse ID': mouse_ids
    })
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create box plot
    sns.boxplot(data=df, x='Mouse ID', y='Speed (cm/s)', ax=ax)
    
    # Customize the plot
    ax.set_title(f'Mouse Speed Distribution', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Mouse ID', fontsize=14, fontweight='bold')
    ax.set_ylabel('Speed (cm/s)', fontsize=14, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='both', labelsize=12)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
    
    return df

def plot_speed_comparison(trajectory_collection, save_path=None):
    """
    Compare speed distributions across different calculation modes.
    
    Args:
        trajectory_collection: TrajectoryCollection object
        save_path: Path to save the plot (optional)
    """
    
    modes = ["linear", "mean", "single"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, mode in enumerate(modes):
        speeds_dict = trajectory_collection.to_speed(window_size=5, mode=mode)
        
        # Prepare data
        plot_data = []
        mouse_ids = []
        
        for mouse_id, speeds in speeds_dict.items():
            if speeds:
                plot_data.extend(speeds)
                mouse_ids.extend([f"Mouse {mouse_id}"] * len(speeds))
        
        df = pd.DataFrame({
            'Speed (cm/s)': plot_data,
            'Mouse ID': mouse_ids
        })
        
        # Create box plot
        sns.boxplot(data=df, x='Mouse ID', y='Speed (cm/s)', ax=axes[i])
        axes[i].set_title(f'{mode.capitalize()} Mode', fontsize=14, fontweight='bold')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Speed Distribution Comparison Across Calculation Modes (Box Plots)', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()

def main():
    """Main function to run the speed analysis."""
    
    # Load trajectory data
    print("Loading trajectory data...")
    trajectory_collection = TrajectoryCollection("data/gt_mtrack.csv", fps=30, scale=0.259)
    print(f"Loaded {len(trajectory_collection)} mice trajectories")
    
    # Plot speed distributions
    print("Creating speed distribution plots...")
    df = plot_speed_distributions(
        trajectory_collection, 
        window_size=10, 
        mode="single",
        save_path="speed_boxplots.png"
    )
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("=" * 50)
    speeds_dict = trajectory_collection.to_speed(window_size=10, mode="single")
    
    for mouse_id, speeds in speeds_dict.items():
        if speeds:
            print(f"Mouse {mouse_id}:")
            print(f"  Mean speed: {np.mean(speeds):.2f} cm/s")
            print(f"  Std speed: {np.std(speeds):.2f} cm/s")
            print(f"  Min speed: {np.min(speeds):.2f} cm/s")
            print(f"  Max speed: {np.max(speeds):.2f} cm/s")
            print(f"  Number of measurements: {len(speeds)}")
            print()

if __name__ == "__main__":
    main()
