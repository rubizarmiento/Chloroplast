import numpy as np
import pyvista as pv
from scipy.spatial import Delaunay
from scipy.interpolate import RBFInterpolator
from matplotlib.path import Path

def cylinder_with_concentric_circles(radius=1.0, height=2.0, radial_resolution=16, 
                                        height_resolution=8, concentric_circles=3, 
                                        center=None, direction=None):
    """
    Generate a simple cylindrical mesh with concentric circles at the top and bottom.

    Parameters
    ----------
    radius : float, default: 1.0
        Radius of the cylinder.
    height : float, default: 2.0
        Height of the cylinder.
    radial_resolution : int, default: 16
        Number of radial segments (sides) of the cylinder.
    height_resolution : int, default: 8
        Number of segments along the height of the cylinder.
    concentric_circles : int, default: 3
        Number of concentric circles at the top and bottom of the cylinder.
    center : array_like, default: [0, 0, 0]
        Center point of the cylinder (middle point along the axis).
    direction : array_like, default: [0, 0, 1]
        Direction vector of the cylinder axis (will be normalized).
    """
    # Default values
    if center is None:
        center = np.array([0.0, 0.0, 0.0])
    else:
        center = np.array(center)
    
    if direction is None:
        direction = np.array([0.0, 0.0, 1.0])
    else:
        direction = np.array(direction)
    
    # Normalize direction vector
    direction = direction / np.linalg.norm(direction)
    
    # Create local coordinate system
    # Find two orthogonal vectors perpendicular to direction
    if abs(direction[2]) < 0.9:  # direction is not too close to z-axis
        u = np.cross(direction, [0, 0, 1])
    else:  # direction is close to z-axis, use x-axis instead
        u = np.cross(direction, [1, 0, 0])
    
    u = u / np.linalg.norm(u)
    v = np.cross(direction, u)
    v = v / np.linalg.norm(v)
    
    vertices = []
    
    # Calculate radii for concentric circles
    circle_radii = np.linspace(0, radius, concentric_circles + 1)[1:]  # Exclude center point
    
    # Create vertices for the cylindrical surface
    for i in range(height_resolution + 1):
        t = i / height_resolution  # Parameter from 0 to 1
        axis_pos = center + direction * (t - 0.5) * height  # Center cylinder on 'center'
        
        for j in range(radial_resolution):
            theta = 2 * np.pi * j / radial_resolution
            
            # Create point in local coordinate system
            local_x = radius * np.cos(theta)
            local_y = radius * np.sin(theta)
            
            # Transform to world coordinates
            world_pos = axis_pos + local_x * u + local_y * v
            vertices.append(world_pos)
    
    # Add cap vertices (top and bottom)
    for t in [0, 1]:  # Bottom (t=0) and top (t=1)
        axis_pos = center + direction * (t - 0.5) * height
        
        # Center point
        vertices.append(axis_pos)
        
        # Concentric circles
        for circle_radius in circle_radii:
            for j in range(radial_resolution):
                theta = 2 * np.pi * j / radial_resolution
                
                # Create point in local coordinate system
                local_x = circle_radius * np.cos(theta)
                local_y = circle_radius * np.sin(theta)
                
                # Transform to world coordinates
                world_pos = axis_pos + local_x * u + local_y * v
                vertices.append(world_pos)
    
    vertices = np.array(vertices)
    faces = []
    
    # Side faces
    for i in range(height_resolution):
        for j in range(radial_resolution):
            curr_base = i * radial_resolution
            next_base = (i + 1) * radial_resolution
            
            curr = curr_base + j
            next_j = curr_base + ((j + 1) % radial_resolution)
            next_i = next_base + j
            next_i_j = next_base + ((j + 1) % radial_resolution)
            
            faces.extend([3, curr, next_j, next_i])
            faces.extend([3, next_j, next_i_j, next_i])
    
    # Cap faces
    surface_vertices = (height_resolution + 1) * radial_resolution
    
    for cap_offset in [0, 1 + len(circle_radii) * radial_resolution]:  # Bottom and top caps
        center_idx = surface_vertices + cap_offset
        
        # Connect center to first ring
        first_ring_start = center_idx + 1
        for j in range(radial_resolution):
            curr = first_ring_start + j
            next_j = first_ring_start + ((j + 1) % radial_resolution)
            
            if cap_offset == 0:  # Bottom cap
                faces.extend([3, center_idx, next_j, curr])
            else:  # Top cap
                faces.extend([3, center_idx, curr, next_j])
        
        # Connect concentric rings
        for ring in range(len(circle_radii) - 1):
            inner_start = center_idx + 1 + ring * radial_resolution
            outer_start = center_idx + 1 + (ring + 1) * radial_resolution
            
            for j in range(radial_resolution):
                inner_curr = inner_start + j
                inner_next = inner_start + ((j + 1) % radial_resolution)
                outer_curr = outer_start + j
                outer_next = outer_start + ((j + 1) % radial_resolution)
                
                if cap_offset == 0:  # Bottom cap
                    faces.extend([3, inner_curr, inner_next, outer_curr])
                    faces.extend([3, inner_next, outer_next, outer_curr])
                else:  # Top cap
                    faces.extend([3, inner_curr, outer_curr, inner_next])
                    faces.extend([3, inner_next, outer_curr, outer_next])
    
    faces = np.array(faces)
    mesh = pv.PolyData(vertices, faces)
    mesh = mesh.clean()
    
    return mesh

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

def create_rectangular_tube(spline_points, width=1.0, height=1.0, width_resolution=4, height_resolution=4, closed=False):
    """
    Create a rectangular tube around a set of spline points.
    
    Parameters
    ----------
    spline_points : np.ndarray
        Array of points defining the centerline path.
        
    width : float, default: 1.0
        Width of the rectangular cross-section.
        
    height : float, default: 1.0
        Height of the rectangular cross-section.
        
    width_resolution : int, default: 4
        Number of points per width side of the rectangular cross-section.
        Higher values create smoother, more detailed cross-sections.
        
    height_resolution : int, default: 4
        Number of points per height side of the rectangular cross-section.
        Higher values create smoother, more detailed cross-sections.
        
    closed : bool, default: False
        Whether the curve is a closed loop.
        
    Returns
    -------
    pyvista.PolyData
        Mesh representing the rectangular tube.
    """
    
    # Use the spline points as provided (no resampling)
    resampled_points = spline_points
    
    # Calculate tangents and normals
    tangents = np.gradient(resampled_points, axis=0)
    
    # Normalize tangents
    tangent_lengths = np.linalg.norm(tangents, axis=1)
    tangent_lengths[tangent_lengths == 0] = 1  # Avoid division by zero
    tangents = tangents / tangent_lengths[:, np.newaxis]
    
    # Create mesh data structures
    tube_points = []
    faces = []
    
    # Generate rectangular cross-section with specified resolution
    half_width = width / 2.0
    half_height = height / 2.0
    
    # Create rectangle points with width_resolution and height_resolution points per side in local 2D coordinates
    rectangle_2d = []
    
    # Bottom side: left to right
    for i in range(width_resolution):
        x = -half_width + (i / (width_resolution - 1)) * width
        rectangle_2d.append([x, -half_height])
    
    # Right side: bottom to top (excluding corners already added)
    for i in range(1, height_resolution):
        y = -half_height + (i / (height_resolution - 1)) * height
        rectangle_2d.append([half_width, y])
    
    # Top side: right to left (excluding corners already added)
    for i in range(1, width_resolution):
        x = half_width - (i / (width_resolution - 1)) * width
        rectangle_2d.append([x, half_height])
    
    # Left side: top to bottom (excluding corners already added)
    for i in range(1, height_resolution - 1):
        y = half_height - (i / (height_resolution - 1)) * height
        rectangle_2d.append([-half_width, y])
    
    rectangle_2d = np.array(rectangle_2d)
    points_per_section = len(rectangle_2d)
    
    # Generate tube mesh
    for i, center_point in enumerate(resampled_points):
        tangent = tangents[i]
        
        # Create local coordinate system at this point
        # Tangent is the Z-axis of local system
        local_z = tangent
        
        # Create a reasonable local X and Y axis
        # Choose an arbitrary vector not parallel to tangent
        if abs(local_z[2]) < 0.9:
            temp_vector = np.array([0, 0, 1])
        else:
            temp_vector = np.array([1, 0, 0])
        
        # Create local X axis (perpendicular to tangent)
        local_x = np.cross(local_z, temp_vector)
        local_x = local_x / np.linalg.norm(local_x)
        
        # Create local Y axis (perpendicular to both)
        local_y = np.cross(local_z, local_x)
        local_y = local_y / np.linalg.norm(local_y)
        
        # Create rotation matrix from local to global coordinates
        rotation_matrix = np.column_stack([local_x, local_y, local_z])
        
        # Transform rectangle points to 3D and orient them
        for local_point_2d in rectangle_2d:
            # Convert 2D rectangle point to 3D local coordinates
            local_point_3d = np.array([local_point_2d[0], local_point_2d[1], 0])
            
            # Rotate to global orientation and translate to center point
            global_point = center_point + np.dot(rotation_matrix, local_point_3d)
            tube_points.append(global_point)
        
        # Create faces between this rectangle and the previous one
        if i > 0:
            for j in range(points_per_section):
                idx1 = (i-1) * points_per_section + j
                idx2 = (i-1) * points_per_section + (j+1) % points_per_section
                idx3 = i * points_per_section + (j+1) % points_per_section
                idx4 = i * points_per_section + j
                
                # Add two triangular faces
                faces.extend([3, idx1, idx2, idx3])
                faces.extend([3, idx1, idx3, idx4])
    
    # Close the loop if requested   
    if closed and len(resampled_points) > 2:
        # Connect last cross-section to first
        for j in range(points_per_section):
            idx1 = (len(resampled_points)-1) * points_per_section + j
            idx2 = (len(resampled_points)-1) * points_per_section + (j+1) % points_per_section
            idx3 = (j+1) % points_per_section
            idx4 = j
            
            faces.extend([3, idx1, idx2, idx3])
            faces.extend([3, idx1, idx3, idx4])

    # Create mesh
    return pv.PolyData(np.array(tube_points), np.array(faces))

def create_periodic_copies(mesh, n_copies_x_positive=1, n_copies_x_negative=1,
                           n_copies_y_positive=1, n_copies_y_negative=1,
                           n_copies_z_positive=1, n_copies_z_negative=1, 
                           x_limits=[], y_limits=[], z_limits=[]):
    """Create periodic copies of a mesh in the specified directions.
    
    Parameters
    ----------
    mesh : pyvista.PolyData or list of pyvista.PolyData
        The mesh(es) to copy.
    n_copies_x_positive : int, default: 1
        Number of copies in the positive x direction.
    n_copies_x_negative : int, default: 1
        Number of copies in the negative x direction.
    n_copies_y_positive : int, default: 1
        Number of copies in the positive y direction.
    n_copies_y_negative : int, default: 1
        Number of copies in the negative y direction.
    n_copies_z_positive : int, default: 1
        Number of copies in the positive z direction.
    n_copies_z_negative : int, default: 1
        Number of copies in the negative z direction.
    x_limits : list, default: []
        List of two floats specifying the x limits [xmin, xmax]. If empty, uses mesh bounds.
    y_limits : list, default: []
        List of two floats specifying the y limits [ymin, ymax]. If empty, uses mesh bounds.
    z_limits : list, default: []
        List of two floats specifying the z limits [zmin, zmax]. If empty, uses mesh bounds.
        
    Returns
    -------
    list of pyvista.PolyData
        List containing the original mesh and all periodic copies.
    """
    # Handle both single mesh and list of meshes
    if isinstance(mesh, list):
        mesh_list = mesh
    else:
        mesh_list = [mesh]
    
    # Calculate dimensions for translation
    if x_limits:
        x_min, x_max = x_limits
        x_size = x_max - x_min
    else:
        # Get bounds from all meshes
        all_bounds = []
        for m in mesh_list:
            all_bounds.append(m.bounds)
        all_bounds = np.array(all_bounds)
        x_min = all_bounds[:, 0].min()  # minimum x
        x_max = all_bounds[:, 1].max()  # maximum x
        x_size = x_max - x_min
    
    if y_limits:
        y_min, y_max = y_limits
        y_size = y_max - y_min
    else:
        y_min = all_bounds[:, 2].min()  # minimum y
        y_max = all_bounds[:, 3].max()  # maximum y
        y_size = y_max - y_min
    
    if z_limits:
        z_min, z_max = z_limits
        z_size = z_max - z_min
    else:
        z_min = all_bounds[:, 4].min()  # minimum z
        z_max = all_bounds[:, 5].max()  # maximum z
        z_size = z_max - z_min
    
    all_meshes = []
    
    # Generate all translation vectors
    for i in range(-n_copies_x_negative, n_copies_x_positive + 1):
        for j in range(-n_copies_y_negative, n_copies_y_positive + 1):
            for k in range(-n_copies_z_negative, n_copies_z_positive + 1):
                
                # Calculate translation vector
                translation = np.array([
                    i * x_size,
                    j * y_size,
                    k * z_size
                ])
                
                # Create copies of all meshes with this translation
                for original_mesh in mesh_list:
                    if i == 0 and j == 0 and k == 0:
                        # Original mesh (no translation)
                        all_meshes.append(original_mesh.copy())
                    else:
                        # Create translated copy using manual translation
                        translated_mesh = original_mesh.copy()
                        translated_mesh.points = translated_mesh.points + translation
                        all_meshes.append(translated_mesh)
    
    return all_meshes


def plot_periodic_copies(periodic_meshes, show_edges=False, window_size=[1600, 600], 
                        background_color='black', opacity=1.0):
    """Plot a list of periodic mesh copies.
    
    Parameters
    ----------
    periodic_meshes : list of pyvista.PolyData
        List of meshes to plot (typically output from create_periodic_copies).
    show_edges : bool, default: False
        Whether to show mesh edges.
    window_size : list, default: [1600, 600]
        Window size for the plotter.
    background_color : str or tuple, default: 'black'
        Background color for the scene.
    opacity : float, default: 1.0
        Opacity for all meshes (0.0 to 1.0).
        
    Returns
    -------
    pyvista.Plotter
        The plotter object used for visualization.
    """
    plotter = pv.Plotter(window_size=window_size)
    plotter.background_color = background_color
    
    for mesh in periodic_meshes:
        # Use material colors if available
        if 'material_color' in mesh.array_names:
            colors = mesh['material_color']
            plotter.add_mesh(mesh, scalars=colors, rgb=True, show_edges=show_edges, 
                           opacity=opacity)
        elif 'face_colors' in mesh.array_names:
            colors = mesh['face_colors']
            plotter.add_mesh(mesh, scalars=colors[:, :3], rgb=True, show_edges=show_edges,
                           opacity=opacity)
        else:
            plotter.add_mesh(mesh, color='lightblue', show_edges=show_edges, 
                           opacity=opacity)
    
    plotter.show()
    return plotter

def fill_nonplanar_loop_with_grid(loop_pts_3d, target_area=1.0):
    """
    Fill a non-planar closed loop with a triangular grid that preserves curvature.
    - loop_pts_3d: (N,3) ordered, closed loop in 3D (no self-intersections).
    - target_area: target area for triangles in the mesh.
    Returns: pv.PolyData (triangular mesh that follows the loop's curvature).
    """
    
    P = np.asarray(loop_pts_3d, dtype=float)
    assert P.ndim == 2 and P.shape[1] == 3 and P.shape[0] >= 3

    # --- best-fit plane (PCA) and local 2D frame for parameterization
    c = P.mean(axis=0)
    X = P - c
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    n = Vt[2] / np.linalg.norm(Vt[2])            # plane normal (smallest variance dir)
    u = Vt[0] / np.linalg.norm(Vt[0])            # first in-plane axis
    v = np.cross(n, u); v /= np.linalg.norm(v)   # second in-plane axis (right-handed)

    # --- project boundary to 2D for parameterization
    pts2d = np.column_stack((X @ u, X @ v))
    poly_path = Path(pts2d, closed=True)

    # --- calculate bounding box area and determine grid density
    min_x, min_y = pts2d.min(axis=0)
    max_x, max_y = pts2d.max(axis=0)
    bbox_area = (max_x - min_x) * (max_y - min_y)
    
    # Estimate number of triangles needed based on target area
    # Assuming roughly 2 triangles per grid cell and some fill factor
    approx_num_triangles = bbox_area / target_area
    grid_cells_needed = int(np.sqrt(approx_num_triangles / 2))
    
    # Calculate density based on aspect ratio
    aspect_ratio = (max_x - min_x) / (max_y - min_y)
    if aspect_ratio > 1:
        density_x = max(5, int(grid_cells_needed * np.sqrt(aspect_ratio)))
        density_y = max(5, int(grid_cells_needed / np.sqrt(aspect_ratio)))
    else:
        density_x = max(5, int(grid_cells_needed / np.sqrt(1/aspect_ratio)))
        density_y = max(5, int(grid_cells_needed * np.sqrt(1/aspect_ratio)))
    
    gx, gy = np.meshgrid(
        np.linspace(min_x, max_x, density_x),
        np.linspace(min_y, max_y, density_y)
    )
    grid_points_2d = np.c_[gx.ravel(), gy.ravel()]

    # --- keep only interior grid points
    inside_mask = poly_path.contains_points(grid_points_2d)
    interior_points_2d = grid_points_2d[inside_mask]

    # --- Create RBF interpolator to map from 2D to 3D curvature
    # Use the boundary points to learn the 3D surface
    rbf_x = RBFInterpolator(pts2d, P[:, 0], kernel='thin_plate_spline', smoothing=0.1)
    rbf_y = RBFInterpolator(pts2d, P[:, 1], kernel='thin_plate_spline', smoothing=0.1)
    rbf_z = RBFInterpolator(pts2d, P[:, 2], kernel='thin_plate_spline', smoothing=0.1)

    # --- interpolate interior points to follow the curvature
    if len(interior_points_2d) > 0:
        interior_x = rbf_x(interior_points_2d)
        interior_y = rbf_y(interior_points_2d)
        interior_z = rbf_z(interior_points_2d)
        interior_points_3d = np.column_stack([interior_x, interior_y, interior_z])
    else:
        interior_points_3d = np.empty((0, 3))

    # --- merge boundary + interpolated interior points
    all_pts_3d = np.vstack([P, interior_points_3d])
    all_pts_2d = np.vstack([pts2d, interior_points_2d])

    # --- triangulate in 2D space for topology
    tri = Delaunay(all_pts_2d)
    centroids = all_pts_2d[tri.simplices].mean(axis=1)
    inside_tri_mask = poly_path.contains_points(centroids)
    simplices = tri.simplices[inside_tri_mask]

    # --- create mesh with 3D points that follow curvature
    faces = np.hstack([[3, *t] for t in simplices]).astype(np.int64)
    mesh = pv.PolyData(all_pts_3d, faces)
    
    # --- print actual average triangle area for verification
    if mesh.n_cells > 0:
        # Use PyVista's built-in method to compute cell areas
        mesh_with_areas = mesh.compute_cell_sizes()
        triangle_areas = mesh_with_areas['Area']
        avg_area = np.mean(triangle_areas)
        print(f"Target area: {target_area:.3f}, Actual average area: {avg_area:.3f}")
    
    return mesh

def line_between_points(p1, p2, n_segments=10):
    """
    Create a line between two points with a specified number of segments.
    
    Parameters:
    p1 (tuple): First point (x1, y1) for 2D or (x1, y1, z1) for 3D.
    p2 (tuple): Second point (x2, y2) for 2D or (x2, y2, z2) for 3D.
    n_segments (int): Number of segments in the line.
    
    Returns:
    np.ndarray: Array of points along the line.
    """
    
    p1 = np.array(p1)
    p2 = np.array(p2)
    
    # Create parameter values from 0 to 1
    t = np.linspace(0, 1, n_segments)
    
    # Generate points along the line using linear interpolation
    points = np.array([p1 + t_val * (p2 - p1) for t_val in t])
    return points


