#!/usr/bin/env python3
"""Test script for analyzing approach events between mouse pairs."""

import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from pawprint.event.approach import ApproachDetector
from pawprint.data import TrajectoryCollection
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def analyze_approach_events(trajectory_collection, init_threshold=40, end_threshold=5, window_size=120):
    """
    Analyze approach events between all mouse pairs.
    
    Args:
        trajectory_collection: TrajectoryCollection object
        init_threshold: Initial distance threshold (cm)
        end_threshold: End distance threshold (cm)
        window_size: Window size for detection
    
    Returns:
        list: List of approach events
    """
    print(f"Detecting approach events for {len(trajectory_collection)} mice...")
    print(f"Init threshold: {init_threshold} cm, End threshold: {end_threshold} cm")
    print(f"Window size: {window_size} frames")
    print("=" * 60)
    
    approach_detector = ApproachDetector(
        trajectory_collection, 
        init_threshold=init_threshold,
        end_threshold=end_threshold,
        window_size=window_size
    )
    
    events = approach_detector.detect()
    print(f"Detected {len(events)} approach events")
    
    return events

def create_approach_matrix(events, n_mice):
    """
    Create approach count matrix from events.
    
    Args:
        events: List of approach events
        n_mice: Number of mice
    
    Returns:
        tuple: (approach_matrix, object_approach_counts)
    """
    # Initialize matrices
    approach_matrix = np.zeros((n_mice, n_mice))
    object_approach_counts = np.zeros(n_mice)
    
    # Count approach events
    for event in events:
        subject_idx = event.subject_identity - 1
        object_idx = event.object_identity - 1
        approach_matrix[subject_idx, object_idx] += 1
        object_approach_counts[object_idx] += 1
    
    return approach_matrix, object_approach_counts

def plot_approach_matrix(approach_matrix, identities, save_path=None):
    """
    Plot approach count matrix as a heatmap.
    
    Args:
        approach_matrix: 2D array of approach counts
        identities: List of mouse identities
        save_path: Path to save the plot (optional)
    """
    n_mice = len(identities)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(approach_matrix, cmap='YlOrRd', aspect='auto')
    
    # Customize the plot
    ax.set_title('Mouse Pair Approach Count Matrix\n(Number of approach events)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Object Mouse ID', fontsize=14, fontweight='bold')
    ax.set_ylabel('Subject Mouse ID', fontsize=14, fontweight='bold')
    
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
    cbar.set_label('Number of Approach Events', fontsize=12, fontweight='bold')
    
    # Add text annotations
    for i in range(n_mice):
        for j in range(n_mice):
            if i != j:  # Don't annotate diagonal
                text = ax.text(j, i, f'{int(approach_matrix[i, j])}',
                              ha="center", va="center", color="black", fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Matrix plot saved to {save_path}")
    
    plt.show()

def plot_object_approach_counts(object_approach_counts, identities, save_path=None):
    """
    Plot approach counts for each mouse as object.
    
    Args:
        object_approach_counts: Array of approach counts for each mouse
        identities: List of mouse identities
        save_path: Path to save the plot (optional)
    """
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar plot
    mouse_labels = [f'Mouse {id}' for id in identities]
    bars = ax.bar(mouse_labels, object_approach_counts, color='skyblue', alpha=0.7)
    
    # Customize the plot
    ax.set_title('Number of Times Each Mouse Was Approached', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Mouse ID', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Approach Events', fontsize=14, fontweight='bold')
    
    # Rotate x-axis labels
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='both', labelsize=12)
    
    # Add value labels on bars
    for bar, count in zip(bars, object_approach_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(count)}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Object approach plot saved to {save_path}")
    
    plt.show()

def print_summary_statistics(events, approach_matrix, object_approach_counts):
    """
    Print summary statistics for approach events.
    
    Args:
        events: List of approach events
        approach_matrix: 2D array of approach counts
        object_approach_counts: Array of approach counts for each mouse
    """
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print(f"Total approach events: {len(events)}")
    
    if events:
        durations = [event.end_frame - event.start_frame for event in events]
        print(f"Average event duration: {np.mean(durations):.1f} ± {np.std(durations):.1f} frames")
        print(f"Median event duration: {np.median(durations):.1f} frames")
        print(f"Min event duration: {np.min(durations)} frames")
        print(f"Max event duration: {np.max(durations)} frames")
    
    print(f"\nTotal approach interactions: {np.sum(approach_matrix)}")
    print(f"Average approaches per pair: {np.mean(approach_matrix[approach_matrix > 0]):.2f}")
    
    # Find most and least approached mice
    most_approached_idx = np.argmax(object_approach_counts)
    least_approached_idx = np.argmin(object_approach_counts)
    
    print(f"\nMost approached mouse: Mouse {most_approached_idx + 1} ({int(object_approach_counts[most_approached_idx])} times)")
    print(f"Least approached mouse: Mouse {least_approached_idx + 1} ({int(object_approach_counts[least_approached_idx])} times)")
    
    # Find most and least active mice (as subjects)
    subject_counts = np.sum(approach_matrix, axis=1)
    most_active_idx = np.argmax(subject_counts)
    least_active_idx = np.argmin(subject_counts)
    
    print(f"\nMost active mouse (as subject): Mouse {most_active_idx + 1} ({int(subject_counts[most_active_idx])} approaches)")
    print(f"Least active mouse (as subject): Mouse {least_active_idx + 1} ({int(subject_counts[least_active_idx])} approaches)")

def main():
    """Main function to run the approach event analysis."""
    
    # Configuration
    output_file = "data/approach_events.json"
    init_threshold = 40
    end_threshold = 5
    window_size = 120
    
    # Load trajectory data
    print("Loading trajectory data...")
    trajectory_collection = TrajectoryCollection("data/gt_mtrack.csv", fps=30, scale=0.259)
    print(f"Loaded {len(trajectory_collection)} mice trajectories")
    
    # Detect approach events
    events = analyze_approach_events(
        trajectory_collection, 
        init_threshold=init_threshold,
        end_threshold=end_threshold,
        window_size=window_size
    )
    
    # Save events to JSON
    result = [event.to_dict() for event in events]
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Events saved to {output_file}")
    
    # Create approach matrices
    n_mice = len(trajectory_collection)
    approach_matrix, object_approach_counts = create_approach_matrix(events, n_mice)
    
    # Create plots
    print("\nCreating visualization plots...")
    plot_approach_matrix(
        approach_matrix, 
        trajectory_collection.identities,
        save_path="approach_matrix.png"
    )
    
    plot_object_approach_counts(
        object_approach_counts,
        trajectory_collection.identities,
        save_path="object_approach_counts.png"
    )
    
    # Print summary statistics
    print_summary_statistics(events, approach_matrix, object_approach_counts)
    
    # Save data to CSV
    df_events = pd.DataFrame([event.to_dict() for event in events])
    df_events.to_csv('approach_events_data.csv', index=False)
    print(f"\nEvent data saved to approach_events_data.csv")
    
    # Save matrix data
    df_matrix = pd.DataFrame(
        approach_matrix, 
        index=[f'Mouse {id}' for id in trajectory_collection.identities],
        columns=[f'Mouse {id}' for id in trajectory_collection.identities]
    )
    df_matrix.to_csv('approach_matrix_data.csv')
    print(f"Matrix data saved to approach_matrix_data.csv")

if __name__ == "__main__":
    main()