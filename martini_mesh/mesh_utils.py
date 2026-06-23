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

