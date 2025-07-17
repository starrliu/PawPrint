#!/usr/bin/env python3
"""Test script for analyzing proximity time between mouse pairs."""

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

def analyze_proximity_time(trajectory_collection, threshold=5.0, fps=30):
    """
    Analyze proximity time between all mouse pairs.
    
    Args:
        trajectory_collection: TrajectoryCollection object
        threshold: Distance threshold for proximity (cm)
        fps: Frames per second
    
    Returns:
        dict: Dictionary with proximity time data
    """
    identities = trajectory_collection.identities
    proximity_data = {}
    
    print(f"Analyzing proximity time for {len(identities)} mice...")
    print(f"Threshold: {threshold} cm, FPS: {fps}")
    print("=" * 60)
    
    for i, first_id in enumerate(identities):
        for second_id in identities[i+1:]:
            # Get proximity data for this pair
            proximity_list = trajectory_collection.to_proximity(first_id, second_id, threshold)
            
            # Calculate total proximity time
            proximity_frames = sum(1 for x in proximity_list if x == 1)  # Count frames where proximity=1
            proximity_time = proximity_frames / fps  # Convert to seconds
            
            # Calculate percentage of time in proximity
            total_frames = len(proximity_list)
            proximity_percentage = (proximity_frames / total_frames) * 100
            
            # Store data
            pair_key = f"{first_id}-{second_id}"
            proximity_data[pair_key] = {
                'mouse1': first_id,
                'mouse2': second_id,
                'proximity_frames': proximity_frames,
                'proximity_time_seconds': proximity_time,
                'proximity_percentage': proximity_percentage,
                'total_frames': total_frames
            }
            
            print(f"Pair {pair_key}: {proximity_time:.2f}s ({proximity_percentage:.2f}%)")
    
    return proximity_data

def plot_proximity_time_matrix(proximity_data, identities, save_path=None):
    """
    Plot proximity time as a heatmap matrix.
    
    Args:
        proximity_data: Dictionary with proximity time data
        identities: List of mouse identities
        save_path: Path to save the plot (optional)
    """
    n_mice = len(identities)
    matrix = np.zeros((n_mice, n_mice))
    
    # Fill the matrix with proximity times
    for pair_key, data in proximity_data.items():
        mouse1_idx = data['mouse1'] - 1  # Convert to 0-based index
        mouse2_idx = data['mouse2'] - 1
        matrix[mouse1_idx, mouse2_idx] = data['proximity_time_seconds']
        matrix[mouse2_idx, mouse1_idx] = data['proximity_time_seconds']  # Symmetric
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    
    # Customize the plot
    ax.set_title('Mouse Pair Proximity Time Matrix\n(Total time in seconds)', 
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
    cbar.set_label('Proximity Time (seconds)', fontsize=12, fontweight='bold')
    
    # Add text annotations
    for i in range(n_mice):
        for j in range(n_mice):
            if i != j:  # Don't annotate diagonal
                text = ax.text(j, i, f'{matrix[i, j]:.1f}s',
                              ha="center", va="center", color="black", fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Matrix plot saved to {save_path}")
    
    plt.show()

def plot_proximity_percentage_bar(proximity_data, save_path=None):
    """
    Plot proximity percentage as a bar chart.
    
    Args:
        proximity_data: Dictionary with proximity time data
        save_path: Path to save the plot (optional)
    """
    # Prepare data
    pairs = list(proximity_data.keys())
    percentages = [data['proximity_percentage'] for data in proximity_data.values()]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bar plot
    bars = ax.bar(pairs, percentages, color='skyblue', alpha=0.7)
    
    # Customize the plot
    ax.set_title('Proximity Time Percentage by Mouse Pair', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Mouse Pair', fontsize=14, fontweight='bold')
    ax.set_ylabel('Proximity Time (%)', fontsize=14, fontweight='bold')
    
    # Rotate x-axis labels
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='both', labelsize=12)
    
    # Add value labels on bars
    for bar, percentage in zip(bars, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{percentage:.2f}%', ha='center', va='bottom', fontsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Percentage plot saved to {save_path}")
    
    plt.show()

def print_summary_statistics(proximity_data):
    """
    Print summary statistics for proximity time analysis.
    
    Args:
        proximity_data: Dictionary with proximity time data
    """
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Calculate statistics
    proximity_times = [data['proximity_time_seconds'] for data in proximity_data.values()]
    proximity_percentages = [data['proximity_percentage'] for data in proximity_data.values()]
    
    print(f"Total mouse pairs analyzed: {len(proximity_data)}")
    print(f"Average proximity time: {np.mean(proximity_times):.2f} ± {np.std(proximity_times):.2f} seconds")
    print(f"Median proximity time: {np.median(proximity_times):.2f} seconds")
    print(f"Min proximity time: {np.min(proximity_times):.2f} seconds")
    print(f"Max proximity time: {np.max(proximity_times):.2f} seconds")
    print()
    print(f"Average proximity percentage: {np.mean(proximity_percentages):.2f} ± {np.std(proximity_percentages):.2f}%")
    print(f"Median proximity percentage: {np.median(proximity_percentages):.2f}%")
    
    # Find most and least social pairs
    most_social = max(proximity_data.items(), key=lambda x: x[1]['proximity_time_seconds'])
    least_social = min(proximity_data.items(), key=lambda x: x[1]['proximity_time_seconds'])
    
    print(f"\nMost social pair: {most_social[0]} ({most_social[1]['proximity_time_seconds']:.2f}s)")
    print(f"Least social pair: {least_social[0]} ({least_social[1]['proximity_time_seconds']:.2f}s)")

def main():
    """Main function to run the proximity time analysis."""
    
    # Load trajectory data
    print("Loading trajectory data...")
    trajectory_collection = TrajectoryCollection("data/gt_mtrack.csv", fps=30, scale=0.259)
    print(f"Loaded {len(trajectory_collection)} mice trajectories")
    
    # Analyze proximity time
    proximity_data = analyze_proximity_time(
        trajectory_collection, 
        threshold=5.0,  # 5 cm threshold
        fps=30
    )
    
    # Create plots
    print("\nCreating plots...")
    plot_proximity_time_matrix(
        proximity_data, 
        trajectory_collection.identities,
        save_path="proximity_time_matrix.png"
    )
    
    plot_proximity_percentage_bar(
        proximity_data,
        save_path="proximity_percentage.png"
    )
    
    # Print summary statistics
    print_summary_statistics(proximity_data)
    
    # Save data to CSV
    df = pd.DataFrame.from_dict(proximity_data, orient='index')
    df.to_csv('proximity_time_data.csv', index=True)
    print(f"\nData saved to proximity_time_data.csv")

if __name__ == "__main__":
    main()
