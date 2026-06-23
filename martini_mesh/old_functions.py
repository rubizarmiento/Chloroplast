def area_cylinder(radius, height):
    """
    Calculate the surface area and lateral area of a cylinder.
    
    Parameters:
    radius (float): Radius of the cylinder
    height (float): Height of the cylinder
    
    Returns:
    tuple: (lateral_area, total_surface_area)
    """
    lateral_area = 2 * np.pi * radius * height
    base_area = np.pi * radius**2
    total_surface_area = lateral_area + 2 * base_area
    return total_surface_area

def volume_cylinder(radius, height):
    """
    Calculate the volume of a cylinder.
    
    Parameters:
    radius (float): Radius of the cylinder
    height (float): Height of the cylinder
    
    Returns:
    float: Volume of the cylinder
    """
    return np.pi * radius**2 * height

def print_faces_area(mesh):
    """
    Print the statistics of the faces' area of a mesh.
    Parameters
    ----------
    mesh : pyvista.PolyData
        The mesh to analyze.
    Returns
    -------
    area : np.ndarray
        The area of the faces.
    """
    area = mesh.compute_cell_sizes().cell_data['Area']
    print(f"Number of faces: {area.size}")
    print(f"Mean area: {area.mean()}")
    print(f"Std area: {area.std()}")
    print(f"Min area: {area.min()}")
    print(f"Max area: {area.max()}")
    return area

def plot_area_histogram(list_area):
    """
    Compare the area of the faces of two meshes

    Parameters
    ----------
    list_area : list
        List of arrays containing the areas of the faces of two meshes
    ----------
    Returns
    -------
    None
    """
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create histogram for each mesh
    for i, area in enumerate(list_area):
        ax.hist(area, bins=100, alpha=0.5, label=f'Mesh {i+1}')

    # Add labels and title
    ax.set_xlabel('Area')
    ax.set_ylabel('Frequency')
    ax.set_title('Area Histogram')
    ax.legend()

    # Show the plot
    plt.show()

def remesh_surface(mesh, n_points=20000, subdivision=1):
    """
    Remesh a surface to have more uniform triangle distribution using ACVD algorithm.
    
    Parameters
    ----------
    mesh : pyvista.PolyData
        Input mesh to be remeshed.
    
    n_points : int, default: 20000
        Target number of points for the remeshed surface.
    
    subdivision : int, default: 1
        Number of subdivisions to perform before clustering.
        Higher values create more input points for potentially better results.
    
    Returns
    -------
    pyvista.PolyData
        Remeshed surface with more uniform triangle distribution.
    
    Notes
    -----
    This function uses the Approximated Centroidal Voronoi Diagrams (ACVD)
    algorithm to create a more uniform triangulation of the input mesh.
    
    Examples
    --------
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> # Download example mesh
    >>> mesh = examples.download_cow()
    >>> # Remesh with 5000 points
    >>> uniform_mesh = remesh_surface(mesh, n_points=5000)
    """
    import pyacvd
    
    # Create the clustering object
    clus = pyacvd.Clustering(mesh)
    
    # Subdivide mesh for better clustering results if requested
    if subdivision > 1:
        clus.subdivide(subdivision)
    
    # Generate clusters
    clus.cluster(n_points)
    
    # Generate and return remeshed surface
    remeshed = clus.create_mesh()
    
    return remeshed

def simple_vertical_helix(height=10, radius=5, turns=3, n_points=100, handedness="right"):
    """
    Generate a simple vertical helix along the z-axis with specified handedness.

    Parameters
    ----------
    height : float, default: 10
        Total height of the helix.
        
    radius : float, default: 5
        Radius of the helix.
        
    turns : int or float, default: 3
        Number of complete turns in the helix.
        
    n_points : int, default: 100
        Number of points to generate along the helix.

    handedness : str, default: "right"
        Handedness of the helix. Must be either "right" or "left".
        Right-handed helix turns clockwise as it moves away from the viewer.
        Left-handed helix turns counter-clockwise as it moves away from the viewer.
    
    Returns
    -------
    np.ndarray
        Array of shape (n_points, 3) containing the (x, y, z) coordinates 
        of points along the helix.
        
    Examples
    --------
    >>> # Create a right-handed helix (default)
    >>> points_right = simple_vertical_helix(height=10, radius=5, turns=3)
    >>> 
    >>> # Create a left-handed helix
    >>> points_left = simple_vertical_helix(height=10, radius=5, turns=3, handedness="left")
    """
    
    # Validate handedness parameter
    if handedness.lower() not in ["right", "left"]:
        raise ValueError("handedness must be either 'right' or 'left'")
    
    # Set handedness factor (1 for right-handed, -1 for left-handed)
    chirality = 1 if handedness.lower() == "right" else -1
    
    # Generate parameter space
    t = np.linspace(0, 1, n_points) #This creates a linear space from 0 to 1 with n_points points, e.g., [0, 0.01, ..., 1]
    theta = 2 * np.pi * turns * t # This creates a linear space from 0 to 2*pi*turns with n_points points e.g., [0, 0.0628, ..., 6.2832] for turns=3
    
    # Create helix coordinates
    x = radius * np.cos(theta) 
    y = chirality * radius * np.sin(theta)  # Chirality affects the y-coordinate
    z = height * t
    
    return np.column_stack((x, y, z)) 
    
def calculate_helix_length(radius=1.0, pitch=2.0, turns=3.0):
    """
    Calculate the exact arc length of a helix analytically.
    
    Parameters
    ----------
    radius : float
        Radius of the helix
    pitch : float
        Vertical distance per complete turn
    turns : float
        Number of complete turns
    
    Returns
    -------
    length : float
        Total arc length of the helix
    """
    # Arc length formula for helix: L = turns * sqrt((2*pi*r)^2 + pitch^2)
    circumference = 2 * np.pi * radius
    length_per_turn = np.sqrt(circumference**2 + pitch**2)
    total_length = turns * length_per_turn
    return total_length

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


def plot_spline_helix(spline_points):
    """
    Plot the spline helix using matplotlib.

    Parameters
    ----------
    spline_points : np.ndarray
        Array of points representing the spline helix.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the spline helix
    ax.plot(spline_points[:, 0], spline_points[:, 1], spline_points[:, 2])
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()

def create_elliptical_tube(spline_points, major_radius=1.0, minor_radius=0.5, n_points=20, 
                           align_to_global=True, closed=False):
    """
    Create a tube with elliptical cross-section around a set of spline points using Bishop frames.
    
    Parameters
    ----------
    spline_points : np.ndarray
        Array of points defining the centerline path.
        
    major_radius : float, default: 1.0
        Major radius of the elliptical cross-section.
        
    minor_radius : float, default: 0.5
        Minor radius of the elliptical cross-section.
        
    n_points : int, default: 20
        Number of points to use for each cross-section.
        
    align_to_global : bool, default: True
        Whether to align the major axis with the global coordinates when possible.
        
    closed : bool, default: False
        Whether the curve is a closed loop.
        
    Returns
    -------
    pyvista.PolyData
        Mesh representing the elliptical tube.
    """
    import numpy as np
    import pyvista as pv
    
    # Calculate tangents and path lengths
    tangents = np.gradient(spline_points, axis=0)
    tangent_magnitudes = np.sqrt(np.sum(tangents**2, axis=1))
    tangents = tangents / tangent_magnitudes[:, np.newaxis]
    
    # Create mesh data structures
    tube_points = []
    faces = []
    
    # Create elliptical cross-section template
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    ellipse = np.column_stack((
        major_radius * np.cos(theta),
        minor_radius * np.sin(theta),
        np.zeros_like(theta)
    ))
    
    # Generate Bishop frame
    # Bishop frame minimizes rotation by removing the torsion component
    
    # Initialize reference frame
    t0 = tangents[0]
    
    # Choose reference vectors
    global_refs = [np.array([0, 0, 1]), np.array([1, 0, 0]), np.array([0, 1, 0])]
    
    # Find a reference vector that's not parallel to t0
    reference = global_refs[0]
    for ref in global_refs:
        if not np.isclose(np.abs(np.dot(t0, ref)), 1.0):
            reference = ref
            break
    
    # Create initial frame
    n0 = np.cross(t0, reference)
    n0 = n0 / np.linalg.norm(n0)
    b0 = np.cross(t0, n0)
    b0 = b0 / np.linalg.norm(b0)
    
    # Store frames
    frames = [(t0, n0, b0)]
    
    # Generate Bishop frames along the curve
    for i in range(1, len(spline_points)):
        t_prev, n_prev, b_prev = frames[-1]
        t_curr = tangents[i]
        
        # Transport the previous frame to current point
        # Compute the rotation from previous tangent to current tangent
        axis = np.cross(t_prev, t_curr)
        axis_norm = np.linalg.norm(axis)
        
        if axis_norm > 1e-10:
            # Rotation needed - tangent direction changed
            axis = axis / axis_norm
            cos_angle = np.dot(t_prev, t_curr)
            # Handle numerical issues
            if cos_angle > 1.0:
                cos_angle = 1.0
            elif cos_angle < -1.0:
                cos_angle = -1.0
            angle = np.arccos(cos_angle)
            
            # Apply Rodrigues rotation formula to n and b
            sin_angle = np.sin(angle)
            n_curr = n_prev * np.cos(angle) + \
                     np.cross(axis, n_prev) * sin_angle + \
                     axis * np.dot(axis, n_prev) * (1 - np.cos(angle))
                     
            b_curr = b_prev * np.cos(angle) + \
                     np.cross(axis, b_prev) * sin_angle + \
                     axis * np.dot(axis, b_prev) * (1 - np.cos(angle))
        else:
            # No rotation needed - tangent direction unchanged
            n_curr = n_prev
            b_curr = b_prev
        
        # Ensure orthogonality and unit length
        n_curr = n_curr - np.dot(n_curr, t_curr) * t_curr
        n_curr = n_curr / np.linalg.norm(n_curr)
        b_curr = np.cross(t_curr, n_curr)
        
        frames.append((t_curr, n_curr, b_curr))
    
    # Generate tube mesh
    for i, point in enumerate(spline_points):
        t, n, b = frames[i]
        
        # Create rotation matrix to orient ellipse
        rotation = np.column_stack((n, b, t))
        
        # Align major axis consistently if requested
        if align_to_global:
            # Try to align with global up when reasonable
            global_up = np.array([0, 0, 1])
            alignment = np.abs(np.dot(t, global_up))
            
            # If the tube is more horizontal than vertical
            if alignment < 0.7:  
                # Project global up onto the plane perpendicular to t
                proj_up = global_up - np.dot(global_up, t) * t
                norm_proj = np.linalg.norm(proj_up)
                
                if norm_proj > 1e-6:
                    # Use this as the new normal
                    n = proj_up / norm_proj
                    b = np.cross(t, n)
                    rotation = np.column_stack((n, b, t))
        
        # Transform ellipse points to this position
        for e in ellipse:
            # Rotate and translate ellipse point
            tube_points.append(point + np.dot(rotation, e))
        
        # Create faces between this circle and the previous one
        if i > 0:
            for j in range(n_points):
                idx1 = (i-1) * n_points + j
                idx2 = (i-1) * n_points + (j+1) % n_points
                idx3 = i * n_points + (j+1) % n_points
                idx4 = i * n_points + j
                
                # Add two triangular faces
                faces.extend([3, idx1, idx2, idx3])
                faces.extend([3, idx1, idx3, idx4])
    
    # Close the loop if requested
    if closed and len(spline_points) > 2:
        # Connect last cross-section to first
        for j in range(n_points):
            idx1 = (len(spline_points)-1) * n_points + j
            idx2 = (len(spline_points)-1) * n_points + (j+1) % n_points
            idx3 = (j+1) % n_points
            idx4 = j
            
            faces.extend([3, idx1, idx2, idx3])
            faces.extend([3, idx1, idx3, idx4])
    
    # Create mesh
    return pv.PolyData(np.array(tube_points), np.array(faces))

def create_rectangular_tube_old(spline_points, width=1.0, height=1.0, n_points=20, closed=False):
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
        
    n_points : int, default: 20
        Number of points to use for each cross-section.
        
    closed : bool, default: False
        Whether the curve is a closed loop.
        
    Returns
    -------
    pyvista.PolyData
        Mesh representing the rectangular tube.
    """
    import numpy as np
    import pyvista as pv
    
    # Calculate tangents and path lengths
    tangents = np.gradient(spline_points, axis=0)
    
    # Create mesh data structures
    tube_points = []
    faces = []
    
    # Generate rectangular cross-section template
    half_width = width / 2.0
    half_height = height / 2.0
    
    rectangle = np.array([
        [-half_width, -half_height, 0],
        [ half_width, -half_height, 0],
        [ half_width,  half_height, 0],
        [-half_width,  half_height, 0]
    ])
    
    # Generate tube mesh
    for i, point in enumerate(spline_points):
        t = tangents[i]
        
        # Create rotation matrix to orient rectangle along tangent
        rotation = np.eye(3)
        
        # Transform rectangle points to this position
        for e in rectangle:
            # Rotate and translate rectangle point
            tube_points.append(point + np.dot(rotation, e))
        
        # Create faces between this rectangle and the previous one
        if i > 0:
            for j in range(4):
                idx1 = (i-1) * 4 + j
                idx2 = (i-1) * 4 + (j+1) % 4
                idx3 = i * 4 + (j+1) % 4
                idx4 = i * 4 + j
                
                # Add two triangular faces
                faces.extend([3, idx1, idx2, idx3])
                faces.extend([3, idx1, idx3, idx4])
    # Close the loop if requested   
    if closed and len(spline_points) > 2:
        # Connect last cross-section to first
        for j in range(4):
            idx1 = (len(spline_points)-1) * 4 + j
            idx2 = (len(spline_points)-1) * 4 + (j+1) % 4
            idx3 = (j+1) % 4
            idx4 = j
            
            faces.extend([3, idx1, idx2, idx3])
            faces.extend([3, idx1, idx3, idx4])

    # Create mesh
    return pv.PolyData(np.array(tube_points), np.array(faces))

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

def load_obj_with_materials(obj_path, mtl_path=None):
    """Load OBJ file and return PyVista mesh with material colors as attributes.
    
    Parameters
    ----------
    obj_path : str
        Path to the OBJ file.
    mtl_path : str, optional
        Path to the MTL file containing material definitions.

    Returns
    -------
    pv_mesh : pyvista.PolyData or list of pyvista.PolyData
        PyVista mesh(es) with material colors as point/cell data attributes.
        Returns single mesh if OBJ contains one object, list if multiple.
    """
    # Parse materials if MTL file provided
    materials = {}
    if mtl_path:
        materials = parse_mtl_file(mtl_path)
    
    # Load mesh with trimesh
    scene = trimesh.load(obj_path)
    
    if isinstance(scene, trimesh.Scene):
        # Handle multiple objects in scene
        meshes = []
        
        for name, mesh in scene.geometry.items():
            # Create PyVista mesh manually to avoid UV issues
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces)
            cells = np.column_stack([np.full(faces.shape[0], 3), faces]).flatten()
            pv_mesh = pv.PolyData(vertices, cells)
            
            # Add material information as mesh attributes
            if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'material'):
                if hasattr(mesh.visual.material, 'name'):
                    material_name = mesh.visual.material.name
                    pv_mesh['material_name'] = [material_name] * pv_mesh.n_points
                    
                    # Add color information if material exists
                    if material_name in materials:
                        diffuse_color = materials[material_name]['diffuse']
                        # Add as point data (RGB values repeated for each point)
                        pv_mesh['material_color'] = np.tile(diffuse_color, (pv_mesh.n_points, 1))
            
            # Add face colors if available from trimesh
            if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'face_colors'):
                if mesh.visual.face_colors is not None:
                    face_colors = np.asarray(mesh.visual.face_colors)
                    if face_colors.max() > 1.0:
                        face_colors = face_colors / 255.0  # Normalize to 0-1
                    pv_mesh['face_colors'] = face_colors
            
            meshes.append(pv_mesh)
        
        return meshes if len(meshes) > 1 else meshes[0]
    
    else:
        # Single mesh
        vertices = np.asarray(scene.vertices)
        faces = np.asarray(scene.faces)
        cells = np.column_stack([np.full(faces.shape[0], 3), faces]).flatten()
        pv_mesh = pv.PolyData(vertices, cells)
        
        # Add material information if available
        if hasattr(scene, 'visual') and hasattr(scene.visual, 'material'):
            if hasattr(scene.visual.material, 'name'):
                material_name = scene.visual.material.name
                pv_mesh['material_name'] = [material_name] * pv_mesh.n_points
                
                if material_name in materials:
                    diffuse_color = materials[material_name]['diffuse']
                    pv_mesh['material_color'] = np.tile(diffuse_color, (pv_mesh.n_points, 1))
        
        # Add face colors if available
        if hasattr(scene, 'visual') and hasattr(scene.visual, 'face_colors'):
            if scene.visual.face_colors is not None:
                face_colors = np.asarray(scene.visual.face_colors)
                if face_colors.max() > 1.0:
                    face_colors = face_colors / 255.0
                pv_mesh['face_colors'] = face_colors
        
        return pv_mesh
    
    def forbidden_cylinder(tree,mesh, center, radius, height):
        """
        Given a point, get all mesh points inside a cylinder around it.

        Parameters
        ----------
        tree : scipy.spatial.cKDTree
            KDTree for fast nearest neighbor search.
        mesh : pyvista.PolyData
            The mesh to analyze.
        center : np.ndarray
            The center point of the cylinder.
        radius : float
            The radius of the cylinder.
        height : float
            The height of the cylinder.
        Returns
        -------
        np.ndarray
            The indices of the points inside the cylinder.
        """
        # Calculate the search radius for pre-filtering
        # This is the radius of a sphere that contains the cylinder
        search_radius = np.sqrt(radius**2 + (height/2)**2)
        
        # Use KD-Tree to find all points within the search radius (fast pre-filtering)
        candidate_indices = tree.query_ball_point(center, search_radius)
        #print(f"Center: {center}")
        #print(f"Candidate indices: {candidate_indices}")

        # If no candidates found, return empty array
        if len(candidate_indices) == 0:
            return np.array([], dtype=int)

        return candidate_indices

def distribute_points_gaussian(mesh, n_points=100, mean_distance=21.2, std_distance=3.1, 
                               max_attempts=10000, forbiden_radius=11, forbiden_height=6, seed=None):
    """
    Distribute points on a mesh surface following a Gaussian nearest-neighbor distance distribution.
    
    Uses a modified Poisson disc sampling approach that targets distances from a Gaussian distribution.
    
    Parameters
    ----------
    mesh : pyvista.PolyData
        Mesh on which to distribute points.
        
    n_points : int, default: 100
        Number of points to distribute.
        
    mean_distance : float, default: 21.2
        Mean of the Gaussian nearest-neighbor distance distribution.
        
    std_distance : float, default: 3.1
        Standard deviation of the Gaussian nearest-neighbor distance distribution.
        
    max_attempts : int, default: 10000
        Maximum attempts to place each point.
        
    seed : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    np.ndarray
        Array of shape (n_points, 3) containing the coordinates of the distributed points.
    """
    import numpy as np
    import pyvista as pv
    from scipy.spatial import cKDTree
    
    #All points tree
    all_points_tree = cKDTree(mesh.points)

    # Set random seed if provided, it will be used in all random operations
    if seed is not None:
        np.random.seed(seed)
    
    # Generate target distances from Gaussian distribution
    target_distances = generate_nn_distances(n_points, mean_distance, std_distance)
    target_distances = np.sort(target_distances)  # Sort for better placement strategy
    
    # Initialize output points array
    points = np.zeros((n_points, 3))
    n_placed = 0
    
    # Place first point randomly on the surface
    random_idx = np.random.randint(mesh.n_points)
    points[0] = mesh.points[random_idx]
    n_placed = 1
    
    # Create KD-Tree for efficient nearest neighbor searches
    # The KD-Tree is a data structure that allows for fast nearest neighbor searches and has the structure of a binary tree
    # It is built from the points we have placed so far
    # We'll update this as we add points
    tree = cKDTree(points[:1]) #Start with the first point
    
    # Create a forbidden cylinder around the first point
    forbidden_points = forbidden_cylinder(all_points_tree, mesh, points[0], forbiden_radius, forbiden_height,
                                          )
    # Try to place remaining points
    for i in range(1, n_points):

        target_dist = target_distances[i-1]  # Use i-1 because we've placed 1 point already
        
        success = False
        for attempt in range(max_attempts):
            # Pick a random point on the mesh
            candidate_idx = np.random.randint(mesh.n_points)

            #Check if the index is in the forbidden points
            if candidate_idx in forbidden_points:
                continue

            candidate_point = mesh.points[candidate_idx]

            # Find distance to nearest existing point
            dist, _ = tree.query(candidate_point.reshape(1, -1))
            nearest_dist = dist[0]
            
            # Accept if it's close enough to our target distance
            # Use a tolerance that gets wider with more attempts
            tolerance = std_distance * (1.0 + attempt * 0.1)
            if abs(nearest_dist - target_dist) <= tolerance:
                points[n_placed] = candidate_point
                n_placed += 1
                
                # Update KD-Tree with new point
                tree = cKDTree(points[:n_placed])
                success = True

                # Create a new forbidden cylinder around the new point
                new_forbidden_points = forbidden_cylinder(all_points_tree, mesh, candidate_point, forbiden_radius, forbiden_height)

                if any(idx in forbidden_points for idx in new_forbidden_points):
                    continue  # Skip this candidate, it would create overlapping forbidden regions
                # Check if the new forbidden points exist, then we shouldn't place it there
                # Add the new forbidden points to the existing ones
                # Remove duplicates
                forbidden_points = np.unique(forbidden_points)
                print(len(forbidden_points))

                break
        
        # If we couldn't place this point after max_attempts, warn but continue
        if not success:
            print(f"Warning: Could not place point {i+1} after {max_attempts} attempts.")
            
            # Place a point anyway - pick a random surface point that's not too close to others
            for fallback_attempt in range(100):
                candidate_idx = np.random.randint(mesh.n_points)
                candidate_point = mesh.points[candidate_idx]
                dist, _ = tree.query(candidate_point.reshape(1, -1))
                
                # Accept if it's not too close to existing points
                if dist[0] > mean_distance * 0.5:
                    points[n_placed] = candidate_point
                    n_placed += 1
                    tree = cKDTree(points[:n_placed])
                    break
    
    # Analyze the resulting distribution
    if n_placed == n_points:
        # Calculate achieved NN distances
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=2)
        achieved_distances = distances[:, 1]  # Second column is distance to nearest non-self neighbor
        
        # Print statistics
        print(f"Target distribution:  mean={mean_distance:.2f}, std={std_distance:.2f}")
        print(f"Achieved distribution: mean={np.mean(achieved_distances):.2f}, std={np.std(achieved_distances):.2f}")
    else:
        print(f"Warning: Only placed {n_placed} out of {n_points} points.")
    
    # Get positions of the forbidden points
    forbidden_points = mesh.points[forbidden_points]

    return points[:n_placed], forbidden_points


def analyze_point_distribution(points, target_mean=21.2, target_std=3.1):
    """
    Analyze the nearest-neighbor distance distribution of a set of points.
    
    Parameters
    ----------
    points : np.ndarray
        Array of shape (n_points, 3) containing point coordinates.
        
    target_mean : float, default: 21.2
        Target mean distance for comparison.
        
    target_std : float, default: 3.1
        Target standard deviation for comparison.
        
    Returns
    -------
    dict
        Dictionary containing distribution statistics and comparison metrics.
    """
    from scipy.spatial import cKDTree
    from scipy import stats
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Calculate nearest neighbor distances
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=2)
    nn_distances = distances[:, 1]  # Distances to nearest non-self neighbors
    
    # Calculate statistics
    actual_mean = np.mean(nn_distances)
    actual_std = np.std(nn_distances)
    
    # Generate target distribution for comparison
    n_points = len(points)
    target_distances = np.random.normal(target_mean, target_std, n_points)
    
    # Perform Kolmogorov-Smirnov test
    ks_statistic, ks_pvalue = stats.kstest(nn_distances, 'norm', 
                                           args=(target_mean, target_std))
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram of distances
    ax1.hist(nn_distances, bins=20, alpha=0.7, label='Actual')
    ax1.hist(target_distances, bins=20, alpha=0.5, label='Target')
    ax1.set_xlabel('Nearest Neighbor Distance')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distance Distribution Comparison')
    ax1.legend()
    
    # QQ plot
    stats.probplot(nn_distances, dist=stats.norm(loc=target_mean, scale=target_std), 
                  plot=ax2)
    ax2.set_title('Q-Q Plot vs. Target Gaussian')
    
    plt.tight_layout()
    
    # Return results
    results = {
        'actual_mean': actual_mean,
        'actual_std': actual_std,
        'target_mean': target_mean,
        'target_std': target_std,
        'ks_statistic': ks_statistic,
        'ks_pvalue': ks_pvalue,
        'nn_distances': nn_distances,
        'figure': fig
    }
    
    return results

def generate_nn_distances(n_points, mean=21.2, std=3.1, seed=None):
    """
    Generate random distances from a Gaussian distribution.

    Parameters
    ----------
    n_points : int
        Number of distances to generate.
        
    mean : float, default: 21.2
        Mean of the Gaussian distribution.
        
    std : float, default: 3.1
        Standard deviation of the Gaussian distribution.
    
    Returns
    -------
    np.ndarray
        Array of shape (n_points,) containing the generated distances.
    """
    #Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    return np.random.normal(loc=mean, scale=std, size=n_points)

def plot_area_histogram(list_area):
    """
    Compare the area of the faces of two meshes

    Parameters
    ----------
    list_area : list
        List of arrays containing the areas of the faces of two meshes
    ----------
    Returns
    -------
    None
    """
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create histogram for each mesh
    for i, area in enumerate(list_area):
        ax.hist(area, bins=100, alpha=0.5, label=f'Mesh {i+1}')

    # Add labels and title
    ax.set_xlabel('Area')
    ax.set_ylabel('Frequency')
    ax.set_title('Area Histogram')
    ax.legend()

    # Show the plot
    plt.show()

def parse_mtl_file(mtl_path):
    """Parse MTL file to extract material colors.
    
    Parameters
    ----------
    mtl_path : str
        Path to the MTL file.

    Returns
    -------
    materials : dict
        Dictionary mapping material names to their diffuse RGB colors.
    """
    materials = {}
    current_material = None
    
    with open(mtl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('newmtl'):
                current_material = line.split()[1]
                materials[current_material] = {}
            elif line.startswith('Kd') and current_material:
                # Diffuse color (RGB values 0-1)
                rgb = [float(x) for x in line.split()[1:4]]
                materials[current_material]['diffuse'] = rgb
            elif line.startswith('Ka') and current_material:
                # Ambient color (RGB values 0-1)
                rgb = [float(x) for x in line.split()[1:4]]
                materials[current_material]['ambient'] = rgb
            elif line.startswith('Ks') and current_material:
                # Specular color (RGB values 0-1)
                rgb = [float(x) for x in line.split()[1:4]]
                materials[current_material]['specular'] = rgb
    
    return materials

def load_obj_with_materials(obj_path, mtl_path=None):
    """Load OBJ file and return PyVista mesh with material colors as attributes.
    
    Parameters
    ----------
    obj_path : str
        Path to the OBJ file.
    mtl_path : str, optional
        Path to the MTL file containing material definitions.

    Returns
    -------
    pv_mesh : pyvista.PolyData or list of pyvista.PolyData
        PyVista mesh(es) with material colors as point/cell data attributes.
        Returns single mesh if OBJ contains one object, list if multiple.
    """
    # Parse materials if MTL file provided
    materials = {}
    if mtl_path:
        materials = parse_mtl_file(mtl_path)
    
    # Load mesh with trimesh
    scene = trimesh.load(obj_path)
    
    if isinstance(scene, trimesh.Scene):
        # Handle multiple objects in scene
        meshes = []
        
        for name, mesh in scene.geometry.items():
            # Create PyVista mesh manually to avoid UV issues
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces)
            cells = np.column_stack([np.full(faces.shape[0], 3), faces]).flatten()
            pv_mesh = pv.PolyData(vertices, cells)
            
            # Add material information as mesh attributes
            if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'material'):
                if hasattr(mesh.visual.material, 'name'):
                    material_name = mesh.visual.material.name
                    pv_mesh['material_name'] = [material_name] * pv_mesh.n_points
                    
                    # Add color information if material exists
                    if material_name in materials:
                        diffuse_color = materials[material_name]['diffuse']
                        # Add as point data (RGB values repeated for each point)
                        pv_mesh['material_color'] = np.tile(diffuse_color, (pv_mesh.n_points, 1))
            
            # Add face colors if available from trimesh
            if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'face_colors'):
                if mesh.visual.face_colors is not None:
                    face_colors = np.asarray(mesh.visual.face_colors)
                    if face_colors.max() > 1.0:
                        face_colors = face_colors / 255.0  # Normalize to 0-1
                    pv_mesh['face_colors'] = face_colors
            
            meshes.append(pv_mesh)
        
        return meshes if len(meshes) > 1 else meshes[0]
    
    else:
        # Single mesh
        vertices = np.asarray(scene.vertices)
        faces = np.asarray(scene.faces)
        cells = np.column_stack([np.full(faces.shape[0], 3), faces]).flatten()
        pv_mesh = pv.PolyData(vertices, cells)
        
        # Add material information if available
        if hasattr(scene, 'visual') and hasattr(scene.visual, 'material'):
            if hasattr(scene.visual.material, 'name'):
                material_name = scene.visual.material.name
                pv_mesh['material_name'] = [material_name] * pv_mesh.n_points
                
                if material_name in materials:
                    diffuse_color = materials[material_name]['diffuse']
                    pv_mesh['material_color'] = np.tile(diffuse_color, (pv_mesh.n_points, 1))
        
        # Add face colors if available
        if hasattr(scene, 'visual') and hasattr(scene.visual, 'face_colors'):
            if scene.visual.face_colors is not None:
                face_colors = np.asarray(scene.visual.face_colors)
                if face_colors.max() > 1.0:
                    face_colors = face_colors / 255.0
                pv_mesh['face_colors'] = face_colors
        
        return pv_mesh

def plot_mesh_with_materials(mesh, show_edges=True, window_size=[800, 600], background_color='white'):
    """Plot mesh with material colors.
    
    Parameters
    ----------
    mesh : pyvista.PolyData or list of pyvista.PolyData
        Mesh(es) to plot.
    show_edges : bool, default: True
        Whether to show mesh edges.
    window_size : list, default: [800, 600]
        Window size for the plotter.
    background_color : str or tuple, default: 'white'
        Background color for the scene.
    """
    plotter = pv.Plotter(window_size=window_size)
    plotter.background_color = background_color
    
    # Handle single mesh or list of meshes
    if not isinstance(mesh, list):
        mesh = [mesh]
    
    for pv_mesh in mesh:
        # Use material colors if available
        if 'material_color' in pv_mesh.array_names:
            colors = pv_mesh['material_color']
            plotter.add_mesh(pv_mesh, scalars=colors, rgb=True, show_edges=show_edges)
        elif 'face_colors' in pv_mesh.array_names:
            colors = pv_mesh['face_colors']
            plotter.add_mesh(pv_mesh, scalars=colors[:, :3], rgb=True, show_edges=show_edges)
        else:
            plotter.add_mesh(pv_mesh, color='lightblue', show_edges=show_edges)
    
    plotter.show()
    return plotter

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
    import pyvista as pv
    import numpy as np
    
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
    import pyvista as pv
    
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

def load_obj_with_materials(obj_path, mtl_path=None):
    """Load OBJ file and return PyVista mesh with material colors as attributes.
    
    Parameters
    ----------
    obj_path : str
        Path to the OBJ file.
    mtl_path : str, optional
        Path to the MTL file containing material definitions.

    Returns
    -------
    list of pyvista.PolyData
        Always returns a list of PyVista mesh(es) with material colors as point/cell data attributes.
    """
    # Parse materials if MTL file provided
    materials = {}
    if mtl_path:
        materials = parse_mtl_file(mtl_path)
    
    # Load mesh with trimesh
    scene = trimesh.load(obj_path)
    
    if isinstance(scene, trimesh.Scene):
        # Handle multiple objects in scene
        meshes = []
        
        for name, mesh in scene.geometry.items():
            # Create PyVista mesh manually to avoid UV issues
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces)
            cells = np.column_stack([np.full(faces.shape[0], 3), faces]).flatten()
            pv_mesh = pv.PolyData(vertices, cells)
            
            # Add material information as mesh attributes
            if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'material'):
                if hasattr(mesh.visual.material, 'name'):
                    material_name = mesh.visual.material.name
                    pv_mesh['material_name'] = [material_name] * pv_mesh.n_points
                    
                    # Add color information if material exists
                    if material_name in materials:
                        diffuse_color = materials[material_name]['diffuse']
                        # Add as point data (RGB values repeated for each point)
                        pv_mesh['material_color'] = np.tile(diffuse_color, (pv_mesh.n_points, 1))
            
            # Add face colors if available from trimesh
            if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'face_colors'):
                if mesh.visual.face_colors is not None:
                    face_colors = np.asarray(mesh.visual.face_colors)
                    if face_colors.max() > 1.0:
                        face_colors = face_colors / 255.0  # Normalize to 0-1
                    pv_mesh['face_colors'] = face_colors
            
            meshes.append(pv_mesh)
        
        return meshes  # Always return list
    
    else:
        # Single mesh - wrap in list
        vertices = np.asarray(scene.vertices)
        faces = np.asarray(scene.faces)
        cells = np.column_stack([np.full(faces.shape[0], 3), faces]).flatten()
        pv_mesh = pv.PolyData(vertices, cells)
        
        # Add material information if available
        if hasattr(scene, 'visual') and hasattr(scene.visual, 'material'):
            if hasattr(scene.visual.material, 'name'):
                material_name = scene.visual.material.name
                pv_mesh['material_name'] = [material_name] * pv_mesh.n_points
                
                if material_name in materials:
                    diffuse_color = materials[material_name]['diffuse']
                    pv_mesh['material_color'] = np.tile(diffuse_color, (pv_mesh.n_points, 1))
        
        # Add face colors if available
        if hasattr(scene, 'visual') and hasattr(scene.visual, 'face_colors'):
            if scene.visual.face_colors is not None:
                face_colors = np.asarray(scene.visual.face_colors)
                if face_colors.max() > 1.0:
                    face_colors = face_colors / 255.0
                pv_mesh['face_colors'] = face_colors
        
        return [pv_mesh]  # Always return list, even for single mesh
    
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



def direction_vector(angle, axis='z', towards='x'):
    """
    Create a direction vector at a given angle from a specified axis.
    
    Parameters
    ----------
    angle : float
        Angle in degrees from the specified axis.
    axis : str, default: 'z'
        Reference axis ('x', 'y', or 'z') from which the angle is measured.
    towards : str, default: 'x'
        Direction to tilt towards ('x', 'y', or 'z'), must be different from axis.
        
    Returns
    -------
    np.ndarray
        Unit direction vector at the specified angle from the axis.
    """
    if axis == towards:
        raise ValueError("axis and towards must be different")
    
    angle_rad = np.radians(angle)
    
    if axis == 'z':
        if towards == 'x':
            return np.array([np.sin(angle_rad), 0, np.cos(angle_rad)])
        elif towards == 'y':
            return np.array([0, np.sin(angle_rad), np.cos(angle_rad)])
    
    elif axis == 'x':
        if towards == 'y':
            return np.array([np.cos(angle_rad), np.sin(angle_rad), 0])
        elif towards == 'z':
            return np.array([np.cos(angle_rad), 0, np.sin(angle_rad)])
    
    elif axis == 'y':
        if towards == 'x':
            return np.array([np.sin(angle_rad), np.cos(angle_rad), 0])
        elif towards == 'z':
            return np.array([0, np.cos(angle_rad), np.sin(angle_rad)])
    
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

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



def direction_vector(angle, axis='z', towards='x'):
    """
    Create a direction vector at a given angle from a specified axis.
    
    Parameters
    ----------
    angle : float
        Angle in degrees from the specified axis.
    axis : str, default: 'z'
        Reference axis ('x', 'y', or 'z') from which the angle is measured.
    towards : str, default: 'x'
        Direction to tilt towards ('x', 'y', or 'z'), must be different from axis.
        
    Returns
    -------
    np.ndarray
        Unit direction vector at the specified angle from the axis.
    """
    if axis == towards:
        raise ValueError("axis and towards must be different")
    
    angle_rad = np.radians(angle)
    
    if axis == 'z':
        if towards == 'x':
            return np.array([np.sin(angle_rad), 0, np.cos(angle_rad)])
        elif towards == 'y':
            return np.array([0, np.sin(angle_rad), np.cos(angle_rad)])
    
    elif axis == 'x':
        if towards == 'y':
            return np.array([np.cos(angle_rad), np.sin(angle_rad), 0])
        elif towards == 'z':
            return np.array([np.cos(angle_rad), 0, np.sin(angle_rad)])
    
    elif axis == 'y':
        if towards == 'x':
            return np.array([np.sin(angle_rad), np.cos(angle_rad), 0])
        elif towards == 'z':
            return np.array([0, np.cos(angle_rad), np.sin(angle_rad)])
    
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    
def mesh_resolution(mesh):
    """
    Compute and print statistics for mesh face areas.
    
    Parameters
    ----------
    mesh : pyvista.PolyData
        The mesh to analyze.
        
    Returns
    -------
    avg_area : float
        Average area of the mesh faces.
    std_area : float
        Standard deviation of the mesh face areas.
    """
    cell_sizes = mesh.compute_cell_sizes()
    avg_area = cell_sizes['Area'].mean()
    std_area = cell_sizes['Area'].std()
    
    print(f"The mesh has {mesh.n_faces} faces.")
    print(f"The average cell size is {avg_area:.2f} nm^2.")
    print(f"The standard deviation of the cell sizes is {std_area:.2f} nm^2.")
    
    return avg_area, std_area

def helix_pitch_from_tilt(radius, tilt_angle_deg):
    """
    Calculate the helix pitch (axial rise per full turn) 
    given its radius and tilt angle above the horizontal.

    Parameters:
        radius (float): Helix radius R.
        tilt_angle_deg (float): Tilt angle α above horizontal, in degrees.

    Returns:
        float: Pitch P (axial rise per full 360° turn).
    """
    alpha = math.radians(tilt_angle_deg)
    return 2 * math.pi * radius * math.tan(alpha)


def helix_radius_from_tilt(pitch, tilt_angle_deg):
    """
    Calculate the helix radius required to achieve a given tilt angle and pitch.

    Parameters:
        pitch (float): Helix pitch P (axial rise per full turn).
        tilt_angle_deg (float): Desired tilt angle α above the horizontal, in degrees.

    Returns:
        float: Required helix radius R.
    """
    # Convert degrees to radians
    alpha = math.radians(tilt_angle_deg)
    # R = P / (2π · tan(α))
    return pitch / (2 * math.pi * math.tan(alpha))

def helix_radius_from_tilt(pitch, tilt_angle_deg):
    """
    Calculate the helix radius required to achieve a given tilt angle and pitch.

    Parameters:
        pitch (float): Helix pitch P (axial rise per full turn).
        tilt_angle_deg (float): Desired tilt angle α above the horizontal, in degrees.

    Returns:
        float: Required helix radius R.
    """
    # Convert degrees to radians
    alpha = math.radians(tilt_angle_deg)
    # R = P / (2π · tan(α))
    return pitch / (2 * math.pi * math.tan(alpha))

def radius_cone_from_tilt(pitch, tilt_angle_deg):
    """
    Calculate the radius of a cone required to achieve a given tilt angle and pitch.

    Parameters:
        pitch (float): Helix pitch P (axial rise per full turn).
        tilt_angle_deg (float): Desired tilt angle α above the horizontal, in degrees.

    Returns:
        float: Required cone radius R.
    """
    # Convert degrees to radians
    alpha = math.radians(tilt_angle_deg)
    # R = P / (2π · tan(α))
    return pitch / (2 * math.pi * math.tan(alpha))

def add_material_labels(mesh, material_id, color_name):
    """Add material ID and color labels to any mesh (transferable function)."""
    # Add material ID as cell data (standard VTK approach)
    mesh.cell_data['material_id'] = np.full(mesh.n_cells, material_id)
    
    # Add color information
    mesh.cell_data['color_name'] = [color_name] * mesh.n_cells
    
    # Add RGB color values for visualization
    if color_name == 'red':
        rgb = [1.0, 0.0, 0.0]
    elif color_name == 'blue':
        rgb = [0.0, 0.0, 1.0]
    elif color_name == 'green':
        rgb = [0.0, 1.0, 0.0]
    elif color_name == 'yellow':
        rgb = [1.0, 1.0, 0.0]
    elif color_name == 'purple':
        rgb = [0.5, 0.0, 0.5]
    else:
        rgb = [0.5, 0.5, 0.5]  # default gray
    
    # Add RGB as point data for smooth visualization
    mesh.point_data['rgb_color'] = np.tile(rgb, (mesh.n_points, 1))
    
    return mesh

def add_cell_index_labels(mesh):
    """Add index labels to the mesh cells for visualization."""
    # Create labels as strings
    labels = [i for i in range(mesh.n_cells)]

    # Add as cell data
    mesh.cell_data['index'] = labels
    
    return mesh

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

def triangulate_with_triangle(loop_points, max_area=None, min_angle=20):
    """
    Triangulate a loop using Triangle library via meshpy
    
    Parameters:
    -----------
    loop_points : array_like
        Nx3 array of loop boundary points
    max_area : float, optional
        Maximum triangle area constraint
    min_angle : float
        Minimum angle constraint (degrees)
    
    Returns:
    --------
    mesh : pyvista.PolyData
        Triangulated surface mesh
    info : dict
        Triangulation statistics
    """
    # Extract 2D points (assume loop is roughly planar)
    points_2d = loop_points[:, :2]
    
    # Create boundary segments (edges between consecutive points)
    n_points = len(points_2d)
    segments = [[i, (i + 1) % n_points] for i in range(n_points)]
    
    # Set up Triangle input
    triangle_input = triangle.MeshInfo()
    triangle_input.set_points(points_2d.tolist())
    triangle_input.set_facets(segments)
    
    # Build triangulation options
    opts = f"pq{min_angle}a"  # p=PSLG, q=quality, a=area constraint
    if max_area is not None:
        opts += f"{max_area}"
    
    print(f"  Triangle options: {opts}")
    
    # Generate mesh
    mesh_output = triangle.build(triangle_input, opts)
    
    # Extract results
    vertices = np.array(mesh_output.points)
    triangles = np.array(mesh_output.elements)
    
    # Convert back to 3D (interpolate Z from boundary)
    vertices_3d = np.zeros((len(vertices), 3))
    vertices_3d[:, :2] = vertices
    
    # Interpolate Z values from original loop
    for i, (x, y) in enumerate(vertices):
        # Find closest boundary point for Z interpolation
        dists = distance.cdist([[x, y]], loop_points[:, :2])[0]
        closest_idx = np.argmin(dists)
        vertices_3d[i, 2] = loop_points[closest_idx, 2]
    
    # Create PyVista mesh
    faces = []
    for tri in triangles:
        faces.extend([3, tri[0], tri[1], tri[2]])
    
    mesh = pv.PolyData(vertices_3d, faces)
    
    return mesh

def find_cicles_intersection(r1, r2, c1, c2):
    """
    Find the intersection points of two circles.
    
    Parameters:
    r1 (float): Radius of the first circle.
    r2 (float): Radius of the second circle.
    c1 (tuple): Center of the first circle (x1, y1).
    c2 (tuple): Center of the second circle (x2, y2).
    
    Returns:
    list: A list of intersection points as tuples (x, y).
    """
    import math
    
    x1, y1 = c1
    x2, y2 = c2
    
    d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    if d > r1 + r2 or d < abs(r1 - r2):
        return []  # No intersection
    
    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h = math.sqrt(r1**2 - a**2)
    
    x0 = x1 + a * (x2 - x1) / d
    y0 = y1 + a * (y2 - y1) / d
    
    intersection_points = []
    
    if h == 0:
        intersection_points.append((x0, y0))  # One point of intersection
    else:
        rx = -(y2 - y1) * (h / d)
        ry = -(x2 - x1) * (h / d)
        
        intersection_points.append((x0 + rx, y0 + ry))
        intersection_points.append((x0 - rx, y0 - ry))
    
    return intersection_points

def get_x_y_coordinantes_circle_angle(r, c1, angle):
    """
    Get the x and y coordinates of a point on a circle given its radius, center, and angle.
    
    Parameters:
    r (float): Radius of the circle.
    c1 (tuple): Center of the circle (x, y).
    angle (float): Angle in degrees.
    
    Returns:
    tuple: Coordinates of the point on the circle (x, y).
    """
    import math
    x = c1[0] + r * math.cos(math.radians(angle))
    y = c1[1] + r * math.sin(math.radians(angle))
    return (x, y)

def tangent_line_circle(c1, c2, r1, r2):
    """
    Calculate the tangent line between two circles.
    
    Parameters:
    c1 (tuple): Center of the first circle (x1, y1).
    c2 (tuple): Center of the second circle (x2, y2).
    r1 (float): Radius of the first circle.
    r2 (float): Radius of the second circle.
    
    Returns:
    tuple: A tuple containing two points that define the tangent line.
    """
    import math
    
    x1, y1 = c1
    x2, y2 = c2
    
    d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    if d < abs(r1 - r2) or d > r1 + r2:
        return None  # No tangent line exists
    
    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h = math.sqrt(r1**2 - a**2)
    
    x0 = x1 + a * (x2 - x1) / d
    y0 = y1 + a * (y2 - y1) / d
    
    rx = -(y2 - y1) * (h / d)
    ry = -(x2 - x1) * (h / d)
    
    point1 = (x0 + rx, y0 + ry)
    point2 = (x0 - rx, y0 - ry)
    
    return point1, point2

def tangent_line_circle_from_point(c1, c2, r1, r2, point):
    """
    Calculate the tangent line from a point to a circle.
    
    Parameters:
    c1 (tuple): Center of the first circle (x1, y1).
    c2 (tuple): Center of the second circle (x2, y2).
    r1 (float): Radius of the first circle.
    r2 (float): Radius of the second circle.
    point (tuple): Point from which the tangent line is drawn.
    
    Returns:
    tuple: A tuple containing two points that define the tangent line.
    """
    import math
    
    x1, y1 = c1
    x2, y2 = c2
    px, py = point
    
    d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    if d < abs(r1 - r2) or d > r1 + r2:
        return None  # No tangent line exists
    
    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h = math.sqrt(r1**2 - a**2)
    
    x0 = x1 + a * (x2 - x1) / d
    y0 = y1 + a * (y2 - y1) / d
    
    rx = -(y2 - y1) * (h / d)
    ry = -(x2 - x1) * (h / d)
    
    point1 = (x0 + rx, y0 + ry)
    point2 = (x0 - rx, y0 - ry)
    
    return point1, point2

def tangent_points_from_circle_point(c1, r1, c2, r2, point_on_c1):
    """
    Calculate the points where tangent lines from a point on circle 1 touch circle 2.
    
    Parameters:
    c1 (tuple): Center of the first circle (x1, y1)
    r1 (float): Radius of the first circle
    c2 (tuple): Center of the second circle (x2, y2) 
    r2 (float): Radius of the second circle
    point_on_c1 (tuple): Point on circle 1 from which tangents are drawn (px, py)
    
    Returns:
    list: Two tangent points on circle 2 as [(x, y), (x, y)]
    """
    import math
    
    x1, y1 = c1
    x2, y2 = c2
    px, py = point_on_c1
    
    # Distance from point to center of circle 2
    d = math.sqrt((x2 - px)**2 + (y2 - py)**2)
    
    # Check if tangents exist
    if d < r2:
        return []  # Point is inside circle 2, no external tangents
    
    # Angle from point to center of circle 2
    angle_to_center = math.atan2(y2 - py, x2 - px)
    
    # Half-angle of the tangent cone
    half_angle = math.asin(r2 / d)
    
    # Two tangent angles
    angle1 = angle_to_center + half_angle
    angle2 = angle_to_center - half_angle
    
    # Distance from point to tangent points
    tangent_distance = math.sqrt(d**2 - r2**2)
    
    # Calculate tangent points on circle 2
    # These are not exactly on the circle, but the touching points
    t1_x = px + tangent_distance * math.cos(angle1)
    t1_y = py + tangent_distance * math.sin(angle1)
    
    t2_x = px + tangent_distance * math.cos(angle2)
    t2_y = py + tangent_distance * math.sin(angle2)
    
    # Project these points onto circle 2 to get exact tangent points
    # Vector from circle 2 center to approximate tangent point
    v1_x, v1_y = t1_x - x2, t1_y - y2
    v1_len = math.sqrt(v1_x**2 + v1_y**2)
    
    v2_x, v2_y = t2_x - x2, t2_y - y2
    v2_len = math.sqrt(v2_x**2 + v2_y**2)
    
    # Normalize and scale to circle radius
    tangent1 = (x2 + r2 * v1_x / v1_len, y2 + r2 * v1_y / v1_len)
    tangent2 = (x2 + r2 * v2_x / v2_len, y2 + r2 * v2_y / v2_len)
    
    return [tangent1, tangent2]

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

def write_mtl_and_update_obj(
    obj_filename,
    mtl_filename,
    material_name,
    color_rgb,
    transparency=1.0,
    shininess=10,
):
    """
    Write a .mtl file and update an existing .obj file to use the material.

    Parameters:
    - obj_filename (str): Path to the .obj file to update.
    - mtl_filename (str): Name of the .mtl file to create (should match what's used in .obj).
    - material_name (str): Name of the material (used in both files).
    - color_rgb (tuple): RGB tuple (values 0–1).
    - transparency (float): 0 (transparent) to 1 (opaque).
    - shininess (int): Specular shininess (0–1000).
    """
    import os

    # Write MTL file
    r, g, b = color_rgb
    mtl_content = f"""newmtl {material_name}
Kd {r:.3f} {g:.3f} {b:.3f}
Ka 0.200 0.200 0.200
Ks 0.000 0.000 0.000
Ns {shininess}
d {transparency}
illum 2
"""
    with open(mtl_filename, "w") as f:
        f.write(mtl_content)
    print(f"MTL file '{mtl_filename}' written.")

    # Modify OBJ file
    with open(obj_filename, "r") as f:
        lines = f.readlines()

    updated_lines = []
    inserted_mtllib = False
    inserted_usemtl = False

    for i, line in enumerate(lines):
        # Insert mtllib at the top if not already present
        if i == 0 and not any(l.startswith("mtllib") for l in lines[:5]):
            updated_lines.append(f"mtllib {os.path.basename(mtl_filename)}\n")
            inserted_mtllib = True

        # Insert usemtl before first vertex or face
        if not inserted_usemtl and (line.startswith("v ") or line.startswith("f ")):
            updated_lines.append(f"usemtl {material_name}\n")
            inserted_usemtl = True

        updated_lines.append(line)

    # If lines were already present, no need to insert again
    if not inserted_mtllib and not any(l.startswith("mtllib") for l in lines):
        updated_lines.insert(0, f"mtllib {os.path.basename(mtl_filename)}\n")
    if not inserted_usemtl and not any(l.startswith("usemtl") for l in lines):
        # Insert usemtl right before geometry starts
        for idx, l in enumerate(updated_lines):
            if l.startswith("v ") or l.startswith("f "):
                updated_lines.insert(idx, f"usemtl {material_name}\n")
                break

    # Write back modified OBJ file
    with open(obj_filename, "w") as f:
        f.writelines(updated_lines)

    print(f"OBJ file '{obj_filename}' updated to use material '{material_name}'.")

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
    import numpy as np
    import pyvista as pv
    
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

