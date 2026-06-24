def merge_meshes(meshes):
    """
    Merge a list of meshes into a single pyvista.PolyData.

    Parameters
    ----------
    meshes : list of pyvista.PolyData
        Meshes to combine.

    Returns
    -------
    pyvista.PolyData
        All meshes merged into one.
    """
    combined = meshes[0].copy()
    for m in meshes[1:]:
        combined += m
    return combined

def center_mesh(mesh, center=None):
    """
    Center a mesh around a specified point.

    Parameters
    ----------
    mesh : pyvista.PolyData
        The mesh to center.
    center : array_like, optional
        [x, y, z] coordinates to center the mesh around. If None, uses the mesh's center.
    
    Returns
    -------
    pyvista.PolyData
        The centered mesh.
    """
    if center is None:
        center = mesh.center
    mesh.points -= center
    return mesh

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
    
    print(f"The mesh has {mesh.n_cells} faces.")
    print(f"The average cell size is {avg_area:.2f} nm^2.")
    print(f"The standard deviation of the cell sizes is {std_area:.2f} nm^2.")
    
    return avg_area, std_area


