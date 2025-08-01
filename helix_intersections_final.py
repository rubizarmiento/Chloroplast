"""
TRANSFERABLE HELIX INTERSECTION MODULE

This module provides a clean, standalone function for finding intersection points
between two 3D helical splines. Can be easily copied to other projects.

Author: Assistant
Date: August 2025
"""

import numpy as np
from scipy.optimize import minimize
import warnings

def find_helix_intersections(helix1_params, helix2_params, 
                           search_range=(-4*np.pi, 4*np.pi), 
                           tolerance=0.1,
                           grid_resolution=20):
    """
    Find intersection points between two 3D helical splines.
    
    This function uses a hybrid grid search + optimization approach to find
    points where two helices come closest to intersecting.
    
    Parameters:
    -----------
    helix1_params : dict
        First helix parameters with keys:
        - 'center': (x, y, z) tuple for helix center
        - 'radius': float, helix radius  
        - 'pitch': float, vertical distance per full revolution
        - 'axis': str, rotation axis ('x', 'y', or 'z')
        
    helix2_params : dict
        Second helix parameters (same format as helix1_params)
        
    search_range : tuple
        (min_t, max_t) parameter range to search for intersections
        
    tolerance : float
        Maximum distance between points to consider as intersection
        
    grid_resolution : int
        Number of grid points per dimension for initial search
        
    Returns:
    --------
    list of dict
        List of intersection dictionaries, each containing:
        - 'point': (x, y, z) tuple of intersection coordinates
        - 't1': parameter value on first helix
        - 't2': parameter value on second helix  
        - 'distance': actual distance between closest points
        
        Results are sorted by distance (best matches first)
        
    Example:
    --------
    >>> helix1 = {'center': (0,0,0), 'radius': 1.0, 'pitch': 2.0, 'axis': 'z'}
    >>> helix2 = {'center': (0,0,0), 'radius': 1.0, 'pitch': 2.0, 'axis': 'x'}
    >>> intersections = find_helix_intersections(helix1, helix2)
    >>> print(f"Found {len(intersections)} intersections")
    """
    
    def helix_point(t, center, radius, pitch, axis='z'):
        """Calculate 3D point on helix at parameter value t."""
        cx, cy, cz = center
        
        if axis == 'z':
            # Helix around z-axis
            x = cx + radius * np.cos(t)
            y = cy + radius * np.sin(t) 
            z = cz + pitch * t / (2 * np.pi)
        elif axis == 'x':
            # Helix around x-axis
            x = cx + pitch * t / (2 * np.pi)
            y = cy + radius * np.cos(t)
            z = cz + radius * np.sin(t)
        elif axis == 'y':
            # Helix around y-axis
            x = cx + radius * np.cos(t)
            y = cy + pitch * t / (2 * np.pi)
            z = cz + radius * np.sin(t)
        else:
            raise ValueError(f"Invalid axis '{axis}'. Must be 'x', 'y', or 'z'")
            
        return np.array([x, y, z])
    
    def distance_between_helices(params):
        """Calculate distance between points on both helices."""
        t1, t2 = params
        try:
            point1 = helix_point(t1, **helix1_params)
            point2 = helix_point(t2, **helix2_params)
            return np.linalg.norm(point1 - point2)
        except:
            return 1e6  # Return large value for invalid parameters
    
    # Step 1: Grid search to find candidate regions
    t_min, t_max = search_range
    t_values = np.linspace(t_min, t_max, grid_resolution)
    
    candidates = []
    for t1 in t_values:
        for t2 in t_values:
            distance = distance_between_helices([t1, t2])
            if distance < tolerance:
                candidates.append((t1, t2, distance))
    
    # Step 2: Refine candidates using optimization
    intersections = []
    
    for t1_init, t2_init, _ in candidates:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Use Nelder-Mead optimization for robustness
                result = minimize(
                    distance_between_helices,
                    [t1_init, t2_init],
                    method='Nelder-Mead',
                    options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 1000}
                )
            
            if result.success and result.fun < tolerance:
                t1_opt, t2_opt = result.x
                
                # Calculate intersection point as midpoint
                point1 = helix_point(t1_opt, **helix1_params)
                point2 = helix_point(t2_opt, **helix2_params)
                intersection_point = (point1 + point2) / 2
                
                # Check for duplicate intersections
                is_duplicate = False
                for existing in intersections:
                    existing_point = np.array(existing['point'])
                    if np.linalg.norm(intersection_point - existing_point) < tolerance:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    intersections.append({
                        'point': tuple(intersection_point),
                        't1': t1_opt,
                        't2': t2_opt,
                        'distance': result.fun
                    })
                    
        except Exception:
            # Skip failed optimizations
            continue
    
    # Step 3: Sort results by distance quality
    intersections.sort(key=lambda x: x['distance'])
    
    return intersections


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

def test_helix_intersections():
    """Test function with example helices."""
    
    print("Testing Helix Intersection Function")
    print("=" * 40)
    
    # Example 1: Perpendicular helices with same center
    helix1 = {
        'center': (0, 0, 0),
        'radius': 1.0,
        'pitch': 1.5,
        'axis': 'z'
    }
    
    helix2 = {
        'center': (0, 0, 0),
        'radius': 1.0, 
        'pitch': 1.5,
        'axis': 'x'
    }
    
    print("\nExample 1: Perpendicular helices")
    print(f"Helix 1: {helix1}")
    print(f"Helix 2: {helix2}")
    
    intersections = find_helix_intersections(
        helix1, helix2,
        search_range=(-3*np.pi, 3*np.pi),
        tolerance=0.2
    )
    
    print(f"\nFound {len(intersections)} intersections:")
    for i, inter in enumerate(intersections):
        x, y, z = inter['point']
        print(f"  {i+1}: ({x:.3f}, {y:.3f}, {z:.3f}) - distance: {inter['distance']:.4f}")
    
    # Example 2: Offset parallel helices
    helix3 = {
        'center': (0, 0, 0),
        'radius': 2.0,
        'pitch': 1.0,
        'axis': 'z'
    }
    
    helix4 = {
        'center': (1.5, 0, 0),
        'radius': 2.0,
        'pitch': 1.0, 
        'axis': 'z'
    }
    
    print(f"\n{'='*40}")
    print("Example 2: Offset parallel helices")
    print(f"Helix 3: {helix3}")
    print(f"Helix 4: {helix4}")
    
    intersections2 = find_helix_intersections(
        helix3, helix4,
        tolerance=0.3
    )
    
    print(f"\nFound {len(intersections2)} intersections:")
    for i, inter in enumerate(intersections2):
        x, y, z = inter['point']
        print(f"  {i+1}: ({x:.3f}, {y:.3f}, {z:.3f}) - distance: {inter['distance']:.4f}")
    
    return intersections, intersections2


def test2_helix_intersections():
    """Test function with example helices."""
    
    print("Testing Helix Intersection Function")
    print("=" * 40)
    
    # Example 1: Perpendicular helices with same center
    helix1 = {
        'center': (0, 0, 0),
        'radius': 1.0,
        'pitch': 1.5,
        'axis': 'z'
    }
    
    helix2 = {
        'center': (0, 0, 0),
        'radius': 1.0, 
        'pitch': 1.5,
        'axis': 'x'
    }
    
    print("\nExample 1: Perpendicular helices")
    print(f"Helix 1: {helix1}")
    print(f"Helix 2: {helix2}")
    
    intersections = find_helix_intersections(
        helix1, helix2,
        search_range=(-3*np.pi, 3*np.pi),
        tolerance=0.2
    )
    
    print(f"\nFound {len(intersections)} intersections:")
    for i, inter in enumerate(intersections):
        x, y, z = inter['point']
        print(f"  {i+1}: ({x:.3f}, {y:.3f}, {z:.3f}) - distance: {inter['distance']:.4f}")
    
    # Example 2: Offset parallel helices
    helix3 = {
        'center': (0, 0, 0),
        'radius': 2.0,
        'pitch': 1.0,
        'axis': 'z'
    }
    
    helix4 = {
        'center': (1.5, 0, 0),
        'radius': 2.0,
        'pitch': 1.0, 
        'axis': 'z'
    }
    
    print(f"\n{'='*40}")
    print("Example 2: Offset parallel helices")
    print(f"Helix 3: {helix3}")
    print(f"Helix 4: {helix4}")
    
    intersections2 = find_helix_intersections(
        helix3, helix4,
        tolerance=0.3
    )
    
    print(f"\nFound {len(intersections2)} intersections:")
    for i, inter in enumerate(intersections2):
        x, y, z = inter['point']
        print(f"  {i+1}: ({x:.3f}, {y:.3f}, {z:.3f}) - distance: {inter['distance']:.4f}")
    
    return intersections, intersections2

if __name__ == "__main__":
    test_helix_intersections2()
