#!/usr/bin/env python3
"""
Simple test script to generate and plot 3D helix splines starting from different positions.
This script creates multiple helices with different starting points and chirality.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate_helix_spline(x0=0, y0=0, z0=0, radius=1.0, pitch=2.0, turns=3.0, 
                         n_points=200, chirality='right'):
    """
    Generate a 3D helix spline starting from (x0, y0, z0).
    
    Parameters
    ----------
    x0, y0, z0 : float
        Starting coordinates
    radius : float
        Radius of the helix
    pitch : float
        Vertical distance per complete turn
    turns : float
        Number of complete turns
    n_points : int
        Number of points along the helix
    chirality : str
        Handedness of the helix ('right' or 'left')
    
    Returns
    -------
    points : np.ndarray
        Array of shape (n_points, 3) containing helix coordinates
    """
    # Parameter t goes from 0 to 2*pi*turns
    t = np.linspace(0, 2 * np.pi * turns, n_points)
    
    # Chirality factor: right-handed (+1) or left-handed (-1)
    chiral_factor = 1 if chirality.lower() == 'right' else -1
    
    # Helix equations
    x = x0 + radius * np.cos(t)
    y = y0 + chiral_factor * radius * np.sin(t)  # Chirality affects y component
    z = z0 + (pitch / (2 * np.pi)) * t
    
    # Stack into points array
    points = np.column_stack([x, y, z])
    return points


def plot_helices_matplotlib():
    """Plot multiple helices using matplotlib."""
    fig = plt.figure(figsize=(15, 10))
    
    # Create subplots for different chirality
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Define different starting positions
    starting_positions = [
        (0, 0, 0),      # Origin
        (3, 0, 0),      # Offset in x
        (0, 3, 0),      # Offset in y
        (0, 0, 3),      # Offset in z
        (2, 2, 1),      # Diagonal offset
    ]
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    # Plot right-handed helices
    for i, (x0, y0, z0) in enumerate(starting_positions):
        points = generate_helix_spline(
            x0=x0, y0=y0, z0=z0,
            radius=1.0, pitch=2.0, turns=2.5,
            n_points=150, chirality='right'
        )
        
        ax1.plot(points[:, 0], points[:, 1], points[:, 2], 
                color=colors[i], linewidth=2, 
                label=f'Start: ({x0}, {y0}, {z0})')
        # Plot actual start point (first point of the helix)
        ax1.scatter(points[0, 0], points[0, 1], points[0, 2], color=colors[i], s=100, marker='o')
        # Plot end point
        ax1.scatter(points[-1, 0], points[-1, 1], points[-1, 2], 
                   color=colors[i], s=100, marker='s')
    
    # Plot left-handed helices
    for i, (x0, y0, z0) in enumerate(starting_positions):
        points = generate_helix_spline(
            x0=x0, y0=y0, z0=z0,
            radius=1.0, pitch=2.0, turns=2.5,
            n_points=150, chirality='left'
        )
        
        ax2.plot(points[:, 0], points[:, 1], points[:, 2], 
                color=colors[i], linewidth=2, 
                label=f'Start: ({x0}, {y0}, {z0})')
        # Plot actual start point (first point of the helix)
        ax2.scatter(points[0, 0], points[0, 1], points[0, 2], color=colors[i], s=100, marker='o')
        # Plot end point
        ax2.scatter(points[-1, 0], points[-1, 1], points[-1, 2], 
                   color=colors[i], s=100, marker='s')
    
    # Customize the plots
    for ax, title in zip([ax1, ax2], ['Right-Handed Helices', 'Left-Handed Helices']):
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_box_aspect([1,1,1])
    
    plt.tight_layout()
    plt.show()


def compare_chirality():
    """Compare right-handed vs left-handed helices at the same position."""
    fig = plt.figure(figsize=(12, 5))
    
    # Generate both helices at the same starting position
    right_helix = generate_helix_spline(0, 0, 0, radius=1.0, pitch=2.0, turns=3, 
                                       n_points=200, chirality='right')
    left_helix = generate_helix_spline(0, 0, 0, radius=1.0, pitch=2.0, turns=3, 
                                      n_points=200, chirality='left')
    
    # Side-by-side comparison
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Right-handed helix
    ax1.plot(right_helix[:, 0], right_helix[:, 1], right_helix[:, 2], 
            'red', linewidth=3, label='Right-handed')
    ax1.scatter(right_helix[0, 0], right_helix[0, 1], right_helix[0, 2], 
               color='green', s=100, marker='o', label='Start')
    ax1.set_title('Right-Handed Helix')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    
    # Left-handed helix
    ax2.plot(left_helix[:, 0], left_helix[:, 1], left_helix[:, 2], 
            'blue', linewidth=3, label='Left-handed')
    ax2.scatter(left_helix[0, 0], left_helix[0, 1], left_helix[0, 2], 
               color='green', s=100, marker='o', label='Start')
    ax2.set_title('Left-Handed Helix')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()
    
    # Set same viewing angle for comparison
    for ax in [ax1, ax2]:
        ax.view_init(elev=20, azim=45)
        ax.set_box_aspect([1,1,1])
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("3D Helix Spline Test Script")
    print("=" * 40)
    
    # Test 1: Multiple helices with different starting positions and chirality
    print("1. Plotting helices with different starting positions and chirality...")
    plot_helices_matplotlib()
    
    # Test 2: Generate sample helix data with different chiralities
    print("2. Generating sample helix data...")
    
    # Right-handed helix
    right_helix = generate_helix_spline(x0=1, y0=2, z0=3, radius=1.5, pitch=3.0, 
                                       turns=2, chirality='right')
    print(f"Right-handed helix: {len(right_helix)} points")
    print(f"  Start point: {right_helix[0]}")
    print(f"  End point: {right_helix[-1]}")
    print(f"  Length: {np.sum(np.linalg.norm(np.diff(right_helix, axis=0), axis=1)):.3f}")
    
    # Left-handed helix
    left_helix = generate_helix_spline(x0=1, y0=2, z0=3, radius=1.5, pitch=3.0, 
                                      turns=2, chirality='left')
    print(f"Left-handed helix: {len(left_helix)} points")
    print(f"  Start point: {left_helix[0]}")
    print(f"  End point: {left_helix[-1]}")
    print(f"  Length: {np.sum(np.linalg.norm(np.diff(left_helix, axis=0), axis=1)):.3f}")
    
    # Test 3: Show chirality effect on coordinates
    print("3. Demonstrating chirality effect...")
    t_sample = np.pi/4  # 45 degrees
    right_point = generate_helix_spline(0, 0, 0, 1, 2, 1, 50, 'right')[12]  # Sample point
    left_point = generate_helix_spline(0, 0, 0, 1, 2, 1, 50, 'left')[12]   # Same sample point
    
    print(f"At t≈π/4:")
    print(f"  Right-handed: [{right_point[0]:.3f}, {right_point[1]:.3f}, {right_point[2]:.3f}]")
    print(f"  Left-handed:  [{left_point[0]:.3f}, {left_point[1]:.3f}, {left_point[2]:.3f}]")
    print(f"  Y-coordinate difference: {right_point[1] - left_point[1]:.3f}")
    
    print("\nTest complete!")
