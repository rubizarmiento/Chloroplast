"""
Simple Demo: Helix Intersection with Matplotlib Visualization

This is a clean, simple demonstration of the helix intersection function
with 3D matplotlib visualization. Perfect for quick testing and demos.
"""

import sys
sys.path.append('/martini/rubiz/Chloroplast')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from helix_intersections_final import find_helix_intersections

def plot_helices_simple(helix1_params, helix2_params, intersections):
    """
    Simple 3D plot of two helices and their intersection points.
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate helix points for plotting
    t = np.linspace(-3*np.pi, 3*np.pi, 500)
    
    # Helix 1 points
    def helix_coords(t, center, radius, pitch, axis):
        cx, cy, cz = center
        if axis == 'z':
            return (cx + radius * np.cos(t), 
                   cy + radius * np.sin(t), 
                   cz + pitch * t / (2 * np.pi))
        elif axis == 'x':
            return (cx + pitch * t / (2 * np.pi),
                   cy + radius * np.cos(t), 
                   cz + radius * np.sin(t))
        elif axis == 'y':
            return (cx + radius * np.cos(t),
                   cy + pitch * t / (2 * np.pi), 
                   cz + radius * np.sin(t))
    
    # Plot first helix
    x1, y1, z1 = helix_coords(t, **helix1_params)
    ax.plot(x1, y1, z1, 'blue', linewidth=3, alpha=0.8, label=f'Helix 1 ({helix1_params["axis"]}-axis)')
    
    # Plot second helix
    x2, y2, z2 = helix_coords(t, **helix2_params)
    ax.plot(x2, y2, z2, 'red', linewidth=3, alpha=0.8, label=f'Helix 2 ({helix2_params["axis"]}-axis)')
    
    # Plot intersection points
    if intersections:
        coords = np.array([inter['point'] for inter in intersections])
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                  c='gold', s=150, marker='o', edgecolors='black', linewidth=2,
                  label=f'{len(intersections)} Intersections', zorder=10)
        
        # Add numbers to intersection points
        for i, inter in enumerate(intersections):
            x, y, z = inter['point']
            ax.text(x, y, z + 0.2, f'{i+1}', fontsize=12, fontweight='bold',
                   ha='center', va='center', color='darkred')
    
    # Customize plot
    ax.set_xlabel('X', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z', fontsize=12, fontweight='bold')
    ax.set_title('Helix Intersections Demo', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Set good viewing angle
    ax.view_init(elev=25, azim=45)
    
    return fig

def demo():
    """
    Run a simple demonstration of helix intersection detection.
    """
    print("HELIX INTERSECTION DEMO")
    print("="*30)
    
    # Define two intersecting helices
    helix1 = {
        'center': (185, 0, 62.64/2),
        'radius': 45.0,
        'pitch': 62.64,
        'axis': 'z'
    }
    
    helix2 = {
        'center': (0, 0, 0),
        'radius': 155,
        'pitch': 250.57,
        'axis': 'z'
    }
    print(f"Helix 1: center={helix1['center']}, radius={helix1['radius']}, axis={helix1['axis']}")
    print(f"Helix 2: center={helix2['center']}, radius={helix2['radius']}, axis={helix2['axis']}")
    print(f"Pitch: {helix1['pitch']} (both helices)")
    
    # Find intersections
    print(f"\nSearching for intersections...")
    intersections = find_helix_intersections(
        helix1, helix2,
        search_range=(-3*np.pi, 3*np.pi),
        tolerance=0.2,
        grid_resolution=25
    )
    
    print(f"Found {len(intersections)} intersection points:")
    print("-" * 50)
    
    for i, inter in enumerate(intersections):
        x, y, z = inter['point']
        print(f"Point {i+1}: ({x:7.3f}, {y:7.3f}, {z:7.3f})")
        print(f"         Distance: {inter['distance']:.4f}")
        print(f"         Parameters: t1={inter['t1']:6.3f}, t2={inter['t2']:6.3f}")
        print()
    
    # Create visualization
    print("Creating 3D visualization...")
    fig = plot_helices_simple(helix1, helix2, intersections)
    
    # Save the plot
    plt.savefig('/martini/rubiz/Chloroplast/tests/demo_helix_intersections.png', 
                dpi=300, bbox_inches='tight')
    print("Plot saved as: demo_helix_intersections.png")
    
    # Show the plot
    plt.show()
    
    return intersections

if __name__ == "__main__":
    demo()
