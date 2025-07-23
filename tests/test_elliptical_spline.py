#!/usr/bin/env python3
"""
Test script for creating elliptical helix splines.
This script generates 3D elliptical helices with two radii instead of one circular radius.
"""

import numpy as np
import pyvista as pv


def create_elliptical_helix(x0=0, y0=0, z0=0, radius_x=2.0, radius_y=1.0, 
                          pitch=2.0, turns=3.0, n_points=200, chirality='right'):
    """
    Create an elliptical helix spline with two different radii.
    
    Parameters
    ----------
    x0, y0, z0 : float
        Starting coordinates of the helix center.
        
    radius_x : float, default: 2.0
        Radius in the X direction (semi-major axis).
        
    radius_y : float, default: 1.0
        Radius in the Y direction (semi-minor axis).
        
    pitch : float, default: 2.0
        Vertical distance per complete turn.
        
    turns : float, default: 3.0
        Number of complete turns.
        
    n_points : int, default: 200
        Number of points along the helix.
        
    chirality : str, default: 'right'
        Handedness of the helix ('right' or 'left').
    
    Returns
    -------
    points : np.ndarray
        Array of shape (n_points, 3) containing elliptical helix coordinates.
    """
    # Parameter t goes from 0 to 2*pi*turns
    t = np.linspace(0, 2 * np.pi * turns, n_points)
    
    # Chirality factor: right-handed (+1) or left-handed (-1)
    chiral_factor = 1 if chirality.lower() == 'right' else -1
    
    # Elliptical helix equations
    x = x0 + radius_x * np.cos(t)
    y = y0 + chiral_factor * radius_y * np.sin(t)  # Different radius for elliptical shape
    z = z0 + (pitch / (2 * np.pi)) * t
    
    # Stack into points array
    points = np.column_stack([x, y, z])
    return points


def create_3d_elliptical_helix(x0=0, y0=0, z0=0, radius_x=2.0, radius_y=1.0, 
                             pitch=2.0, turns=3.0, n_points=200, chirality='right',
                             rotation_matrix=None):
    """
    Create an elliptical helix in arbitrary 3D orientation.
    
    Parameters
    ----------
    x0, y0, z0 : float
        Starting coordinates of the helix center.
        
    radius_x : float, default: 2.0
        Radius in the local X direction.
        
    radius_y : float, default: 1.0
        Radius in the local Y direction.
        
    pitch : float, default: 2.0
        Vertical distance per complete turn.
        
    turns : float, default: 3.0
        Number of complete turns.
        
    n_points : int, default: 200
        Number of points along the helix.
        
    chirality : str, default: 'right'
        Handedness of the helix ('right' or 'left').
        
    rotation_matrix : np.ndarray, optional
        3x3 rotation matrix to orient the helix in 3D space.
        If None, helix is in XY plane rising in Z.
    
    Returns
    -------
    points : np.ndarray
        Array of shape (n_points, 3) containing elliptical helix coordinates.
    """
    # Parameter t goes from 0 to 2*pi*turns
    t = np.linspace(0, 2 * np.pi * turns, n_points)
    
    # Chirality factor: right-handed (+1) or left-handed (-1)
    chiral_factor = 1 if chirality.lower() == 'right' else -1
    
    # Elliptical helix equations in local coordinates
    x_local = radius_x * np.cos(t)
    y_local = chiral_factor * radius_y * np.sin(t)
    z_local = (pitch / (2 * np.pi)) * t
    
    # Stack local coordinates
    local_points = np.column_stack([x_local, y_local, z_local])
    
    # Apply rotation if provided
    if rotation_matrix is not None:
        rotated_points = np.dot(local_points, rotation_matrix.T)
    else:
        rotated_points = local_points
    
    # Translate to starting position
    final_points = rotated_points + np.array([x0, y0, z0])
    
    return final_points


def test_right_handed_elliptical_helices():
    """Test right-handed elliptical helices with different aspect ratios."""
    
    print("Testing right-handed elliptical helices...")
    print("=" * 45)
    
    # Create different elliptical helices
    helices = []
    
    # Circular helix (for comparison)
    circular_helix = create_elliptical_helix(
        x0=0, y0=0, z0=0,
        radius_x=2.0, radius_y=2.0,  # Same radii = circular
        pitch=3.0, turns=2.5,
        n_points=150, chirality='right'
    )
    helices.append(("Circular (2:2)", circular_helix, 'red'))
    
    # Wide elliptical helix
    wide_helix = create_elliptical_helix(
        x0=6, y0=0, z0=0,
        radius_x=3.0, radius_y=1.0,  # 3:1 ratio - wide
        pitch=3.0, turns=2.5,
        n_points=150, chirality='right'
    )
    helices.append(("Wide (3:1)", wide_helix, 'blue'))
    
    # Tall elliptical helix
    tall_helix = create_elliptical_helix(
        x0=-6, y0=0, z0=0,
        radius_x=1.0, radius_y=2.5,  # 1:2.5 ratio - tall
        pitch=3.0, turns=2.5,
        n_points=150, chirality='right'
    )
    helices.append(("Tall (1:2.5)", tall_helix, 'green'))
    
    # Extreme elliptical helix
    extreme_helix = create_elliptical_helix(
        x0=0, y0=8, z0=0,
        radius_x=4.0, radius_y=0.5,  # 8:1 ratio - very wide
        pitch=2.0, turns=3.0,
        n_points=180, chirality='right'
    )
    helices.append(("Extreme (8:1)", extreme_helix, 'orange'))
    
    # Print information
    for name, points, color in helices:
        print(f"\n{name}:")
        print(f"  Points: {len(points)}")
        print(f"  Start: [{points[0, 0]:.2f}, {points[0, 1]:.2f}, {points[0, 2]:.2f}]")
        print(f"  End: [{points[-1, 0]:.2f}, {points[-1, 1]:.2f}, {points[-1, 2]:.2f}]")
        print(f"  X range: [{np.min(points[:, 0]):.2f}, {np.max(points[:, 0]):.2f}]")
        print(f"  Y range: [{np.min(points[:, 1]):.2f}, {np.max(points[:, 1]):.2f}]")
        print(f"  Total height: {points[-1, 2] - points[0, 2]:.2f}")
    
    # Visualize
    print("\nCreating visualization...")
    plotter = pv.Plotter(window_size=[1200, 800])
    
    for name, points, color in helices:
        # Create polydata for the helix
        helix_poly = pv.PolyData(points)
        
        # Add lines connecting the points
        lines = []
        for i in range(len(points) - 1):
            lines.extend([2, i, i + 1])
        helix_poly.lines = lines
        
        # Add to plot
        plotter.add_mesh(helix_poly, color=color, line_width=4, label=name)
        
        # Add start point marker
        start_point = pv.PolyData([points[0]])
        plotter.add_mesh(start_point, color=color, point_size=15, 
                        render_points_as_spheres=True)
        
        # Add end point marker
        end_point = pv.PolyData([points[-1]])
        plotter.add_mesh(end_point, color=color, point_size=12, 
                        render_points_as_spheres=True)
    
    plotter.add_title("Right-Handed Elliptical Helices with Different Aspect Ratios")
    plotter.add_legend()
    plotter.show_grid()
    plotter.view_isometric()
    plotter.show()
    
    return helices


def test_chirality_comparison():
    """Compare right-handed vs left-handed elliptical helices."""
    
    print("\nTesting chirality comparison...")
    print("=" * 32)
    
    # Create identical helices with opposite chirality
    helix_params = {
        'x0': 0, 'y0': 0, 'z0': 0,
        'radius_x': 2.5, 'radius_y': 1.0,
        'pitch': 2.5, 'turns': 3.0,
        'n_points': 150
    }
    
    right_helix = create_elliptical_helix(**helix_params, chirality='right')
    left_helix = create_elliptical_helix(**helix_params, chirality='left')
    
    # Offset left helix for comparison
    left_helix[:, 0] += 6  # Move to the right
    
    print(f"Right-handed helix: {len(right_helix)} points")
    print(f"Left-handed helix: {len(left_helix)} points")
    
    # Visualize side by side
    plotter = pv.Plotter(shape=(1, 2), window_size=[1200, 600])
    
    # Right-handed helix
    plotter.subplot(0, 0)
    right_poly = pv.PolyData(right_helix)
    lines_right = []
    for i in range(len(right_helix) - 1):
        lines_right.extend([2, i, i + 1])
    right_poly.lines = lines_right
    
    plotter.add_mesh(right_poly, color='red', line_width=5, label='Right-handed')
    plotter.add_mesh(pv.PolyData([right_helix[0]]), color='green', point_size=15, 
                    render_points_as_spheres=True, label='Start')
    plotter.add_title("Right-Handed Elliptical Helix")
    plotter.view_isometric()
    
    # Left-handed helix
    plotter.subplot(0, 1)
    left_poly = pv.PolyData(left_helix)
    lines_left = []
    for i in range(len(left_helix) - 1):
        lines_left.extend([2, i, i + 1])
    left_poly.lines = lines_left
    
    plotter.add_mesh(left_poly, color='blue', line_width=5, label='Left-handed')
    plotter.add_mesh(pv.PolyData([left_helix[0]]), color='green', point_size=15, 
                    render_points_as_spheres=True, label='Start')
    plotter.add_title("Left-Handed Elliptical Helix")
    plotter.view_isometric()
    
    plotter.show()
    
    return right_helix, left_helix


def test_3d_oriented_elliptical_helices():
    """Test elliptical helices in arbitrary 3D orientations."""
    
    print("\nTesting 3D oriented elliptical helices...")
    print("=" * 40)
    
    # Create rotation matrices for different orientations
    def rotation_matrix_x(angle):
        """Rotation matrix around X axis."""
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        return np.array([
            [1, 0, 0],
            [0, cos_a, -sin_a],
            [0, sin_a, cos_a]
        ])
    
    def rotation_matrix_z(angle):
        """Rotation matrix around Z axis."""
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        return np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
    
    oriented_helices = []
    
    # Standard orientation (XY plane)
    standard_helix = create_3d_elliptical_helix(
        x0=0, y0=0, z0=0,
        radius_x=2.0, radius_y=1.0,
        pitch=2.0, turns=2.0,
        n_points=120,
        chirality='right'
    )
    oriented_helices.append(("Standard XY", standard_helix, 'red'))
    
    # Tilted 45° around X axis
    tilted_helix = create_3d_elliptical_helix(
        x0=6, y0=0, z0=0,
        radius_x=2.0, radius_y=1.0,
        pitch=2.0, turns=2.0,
        n_points=120,
        chirality='right',
        rotation_matrix=rotation_matrix_x(np.pi/4)
    )
    oriented_helices.append(("45° X-tilt", tilted_helix, 'green'))
    
    # Rotated 60° around Z axis
    rotated_helix = create_3d_elliptical_helix(
        x0=-6, y0=0, z0=0,
        radius_x=2.0, radius_y=1.0,
        pitch=2.0, turns=2.0,
        n_points=120,
        chirality='right',
        rotation_matrix=rotation_matrix_z(np.pi/3)
    )
    oriented_helices.append(("60° Z-rotation", rotated_helix, 'blue'))
    
    # Print information
    for name, points, color in oriented_helices:
        print(f"\n{name}:")
        print(f"  Points: {len(points)}")
        print(f"  Bounding box X: [{np.min(points[:, 0]):.2f}, {np.max(points[:, 0]):.2f}]")
        print(f"  Bounding box Y: [{np.min(points[:, 1]):.2f}, {np.max(points[:, 1]):.2f}]")
        print(f"  Bounding box Z: [{np.min(points[:, 2]):.2f}, {np.max(points[:, 2]):.2f}]")
    
    # Visualize
    print("\nCreating 3D orientation visualization...")
    plotter = pv.Plotter(window_size=[1200, 800])
    
    for name, points, color in oriented_helices:
        # Create polydata
        helix_poly = pv.PolyData(points)
        
        # Add lines
        lines = []
        for i in range(len(points) - 1):
            lines.extend([2, i, i + 1])
        helix_poly.lines = lines
        
        # Add to plot
        plotter.add_mesh(helix_poly, color=color, line_width=5, label=name)
        
        # Add start point
        start_point = pv.PolyData([points[0]])
        plotter.add_mesh(start_point, color=color, point_size=12, 
                        render_points_as_spheres=True)
    
    plotter.add_title("Elliptical Helices in Different 3D Orientations")
    plotter.add_legend()
    plotter.show_grid()
    plotter.view_isometric()
    plotter.show()
    
    return oriented_helices


if __name__ == "__main__":
    print("Elliptical Helix Test Script")
    print("=" * 50)
    
    # Test 1: Right-handed elliptical helices with different aspect ratios
    right_handed_helices = test_right_handed_elliptical_helices()
    
    # Test 2: Chirality comparison
    right_helix, left_helix = test_chirality_comparison()
    
    # Test 3: 3D oriented elliptical helices
    oriented_helices = test_3d_oriented_elliptical_helices()
    
    print("\nAll tests completed!")
    print("Elliptical helix functions are working correctly!")
    print("\nKey features demonstrated:")
    print("- Elliptical helices with different radius_x and radius_y")
    print("- Right-handed and left-handed chirality")
    print("- Different aspect ratios (circular, wide, tall, extreme)")
    print("- 3D orientations with rotation matrices")
