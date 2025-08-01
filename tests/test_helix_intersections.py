"""
Test script for finding intersections between two 3D helical splines.
This module provides functions to compute intersection points between helices.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, minimize
from mpl_toolkits.mplot3d import Axes3D
import warnings

def helix_point(t, center, radius, pitch, axis='z'):
    """
    Calculate a point on a helix given parameter t.
    
    Parameters:
    -----------
    t : float or array
        Parameter along the helix
    center : tuple or array
        (x, y, z) coordinates of helix center
    radius : float
        Radius of the helix
    pitch : float
        Vertical distance per full revolution
    axis : str
        Axis of rotation ('x', 'y', or 'z')
    
    Returns:
    --------
    array : (3,) or (3, n) array of coordinates
    """
    cx, cy, cz = center
    
    if axis == 'z':
        x = cx + radius * np.cos(t)
        y = cy + radius * np.sin(t)
        z = cz + pitch * t / (2 * np.pi)
    elif axis == 'x':
        x = cx + pitch * t / (2 * np.pi)
        y = cy + radius * np.cos(t)
        z = cz + radius * np.sin(t)
    elif axis == 'y':
        x = cx + radius * np.cos(t)
        y = cy + pitch * t / (2 * np.pi)
        z = cz + radius * np.sin(t)
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")
    
    return np.array([x, y, z])

def distance_between_helices(params, helix1_params, helix2_params):
    """
    Calculate squared distance between two points on different helices.
    
    Parameters:
    -----------
    params : array
        [t1, t2] parameters for both helices
    helix1_params : dict
        Parameters for first helix (center, radius, pitch, axis)
    helix2_params : dict
        Parameters for second helix (center, radius, pitch, axis)
    
    Returns:
    --------
    float : squared distance between points
    """
    t1, t2 = params
    
    point1 = helix_point(t1, **helix1_params)
    point2 = helix_point(t2, **helix2_params)
    
    return np.sum((point1 - point2)**2)

def find_helix_intersections(helix1_params, helix2_params, 
                           t_range=(-10*np.pi, 10*np.pi), 
                           num_initial_guesses=50, 
                           tolerance=1e-6):
    """
    Find intersection points between two 3D helical splines.
    
    Parameters:
    -----------
    helix1_params : dict
        Parameters for first helix: {'center': (x,y,z), 'radius': r, 'pitch': p, 'axis': 'z'}
    helix2_params : dict
        Parameters for second helix: {'center': (x,y,z), 'radius': r, 'pitch': p, 'axis': 'z'}
    t_range : tuple
        Range of parameter values to search
    num_initial_guesses : int
        Number of initial guesses for optimization
    tolerance : float
        Tolerance for considering points as intersections
    
    Returns:
    --------
    intersections : list of dict
        List of intersection points with format:
        [{'point': (x,y,z), 't1': t1_value, 't2': t2_value, 'distance': dist}, ...]
    """
    intersections = []
    t_min, t_max = t_range
    
    # Generate initial guesses
    np.random.seed(42)  # For reproducible results
    initial_guesses = np.random.uniform(t_min, t_max, (num_initial_guesses, 2))
    
    # Add some systematic guesses
    t_systematic = np.linspace(t_min, t_max, int(np.sqrt(num_initial_guesses)))
    for t1 in t_systematic:
        for t2 in t_systematic[:5]:  # Limit systematic combinations
            initial_guesses = np.vstack([initial_guesses, [t1, t2]])
    
    found_intersections = []
    
    for guess in initial_guesses:
        try:
            # Use minimize to find local minimum of distance function
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = minimize(
                    distance_between_helices,
                    guess,
                    args=(helix1_params, helix2_params),
                    method='BFGS',
                    options={'gtol': 1e-8, 'ftol': 1e-12}
                )
            
            if result.success and result.fun < tolerance**2:
                t1_opt, t2_opt = result.x
                
                # Calculate the intersection point
                point1 = helix_point(t1_opt, **helix1_params)
                point2 = helix_point(t2_opt, **helix2_params)
                intersection_point = (point1 + point2) / 2
                distance = np.sqrt(result.fun)
                
                # Check if this intersection is already found
                is_duplicate = False
                for existing in found_intersections:
                    if np.linalg.norm(intersection_point - existing['point']) < tolerance * 10:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    found_intersections.append({
                        'point': intersection_point,
                        't1': t1_opt,
                        't2': t2_opt,
                        'distance': distance
                    })
        
        except:
            continue
    
    # Sort by distance (best intersections first)
    found_intersections.sort(key=lambda x: x['distance'])
    
    return found_intersections

def visualize_helix_intersections(helix1_params, helix2_params, intersections, 
                                 t_range=(-4*np.pi, 4*np.pi), num_points=1000):
    """
    Visualize two helices and their intersection points.
    
    Parameters:
    -----------
    helix1_params : dict
        Parameters for first helix
    helix2_params : dict
        Parameters for second helix
    intersections : list
        List of intersection points from find_helix_intersections
    t_range : tuple
        Range for visualization
    num_points : int
        Number of points to plot for each helix
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate points for visualization
    t_vis = np.linspace(t_range[0], t_range[1], num_points)
    
    # Plot first helix
    points1 = np.array([helix_point(t, **helix1_params) for t in t_vis]).T
    ax.plot(points1[0], points1[1], points1[2], 'b-', label='Helix 1', alpha=0.7)
    
    # Plot second helix
    points2 = np.array([helix_point(t, **helix2_params) for t in t_vis]).T
    ax.plot(points2[0], points2[1], points2[2], 'r-', label='Helix 2', alpha=0.7)
    
    # Plot intersection points
    if intersections:
        intersection_points = np.array([inter['point'] for inter in intersections]).T
        ax.scatter(intersection_points[0], intersection_points[1], intersection_points[2], 
                  c='green', s=100, marker='o', label=f'Intersections ({len(intersections)})')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('Helix Intersections')
    
    plt.tight_layout()
    return fig

def test_helix_intersections():
    """
    Test the helix intersection function with known cases.
    """
    print("Testing Helix Intersection Function")
    print("=" * 40)
    
    # Test Case 1: Two perpendicular helices
    print("\nTest Case 1: Perpendicular helices")
    helix1 = {
        'center': (185, 0, 0),
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
    
    intersections1 = find_helix_intersections(helix1, helix2, 
                                            t_range=(-6*np.pi, 6*np.pi),
                                            num_initial_guesses=100)
    
    print(f"Found {len(intersections1)} intersections")
    for i, inter in enumerate(intersections1[:5]):  # Show first 5
        print(f"  Intersection {i+1}: {inter['point']} (distance: {inter['distance']:.2e})")
    
    ## Test Case 2: Offset parallel helices
    #print("\nTest Case 2: Offset parallel helices")
    #helix3 = {
    #    'center': (0, 0, 0),
    #    'radius': 1.5,
    #    'pitch': 2.0,
    #    'axis': 'z'
    #}
    #
    #helix4 = {
    #    'center': (2, 0, 0.5),
    #    'radius': 1.5,
    #    'pitch': 2.0,
    #    'axis': 'z'
    #}
    #
    #intersections2 = find_helix_intersections(helix3, helix4,
    #                                        t_range=(-4*np.pi, 4*np.pi),
    #                                        num_initial_guesses=80)
    #
    #print(f"Found {len(intersections2)} intersections")
    #for i, inter in enumerate(intersections2[:3]):  # Show first 3
    #    print(f"  Intersection {i+1}: {inter['point']} (distance: {inter['distance']:.2e})")
    #
    ## Visualize results
    #if len(intersections1) > 0:
    #    fig1 = visualize_helix_intersections(helix1, helix2, intersections1)
    #    plt.savefig('/martini/rubiz/Chloroplast/tests/perpendicular_helices.png', dpi=150)
    #    plt.show()
    #
    #if len(intersections2) > 0:
    #    fig2 = visualize_helix_intersections(helix3, helix4, intersections2)
    #    plt.savefig('/martini/rubiz/Chloroplast/tests/offset_helices.png', dpi=150)
    #    plt.show()
    #
    #print("\nTest completed!")
    #return intersections1, intersections2

if __name__ == "__main__":
    # Run the test
    test_helix_intersections()
