#!/usr/bin/env python3
"""
Test script demonstrating the VTK/PyVista community's standard workflow
for merging meshes while preserving material identification.

This follows the common practice of assigning material IDs before boolean operations.
"""

import pyvista as pv
import numpy as np


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
    else:
        rgb = [0.5, 0.5, 0.5]  # default gray
    
    # Add RGB as point data for smooth visualization
    mesh.point_data['rgb_color'] = np.tile(rgb, (mesh.n_points, 1))
    
    return mesh


def create_labeled_sphere(center, radius, material_id, color_name):
    """Create a sphere with material ID and color labels."""
    sphere = pv.Sphere(radius=radius, center=center)
    return add_material_labels(sphere, material_id, color_name)


def create_test_spheres():
    """Create two overlapping spheres with material labels."""
    print("Creating two overlapping spheres with material labels...")
    
    sphere1 = create_labeled_sphere(center=[0, 0, 0], radius=1.0, 
                                   material_id=1, color_name='red')
    sphere2 = create_labeled_sphere(center=[1.5, 0, 0], radius=1.0, 
                                   material_id=2, color_name='blue')
    
    print(f"Sphere 1: {sphere1.n_cells} cells, material_id=1 (red)")
    print(f"Sphere 2: {sphere2.n_cells} cells, material_id=2 (blue)")
    
    return sphere1, sphere2


def perform_merge_operation(mesh1, mesh2):
    """Perform boolean union operation on two meshes."""
    print("\nPerforming boolean union...")
    merged_mesh = mesh1.boolean_union(mesh2)
    print(f"Merged mesh: {merged_mesh.n_cells} cells")
    return merged_mesh


def analyze_material_preservation(merged_mesh, original_meshes):
    """Analyze and handle material preservation in merged mesh."""
    # Check if material information is preserved
    preserved, result = check_material_preservation(merged_mesh)
    
    if preserved:
        print("\n✓ Material IDs preserved in merged mesh!")
        # Extract regions by material
        regions = extract_regions_by_material(merged_mesh)
        return True, regions
        
    else:
        print("\n✗ Material IDs not preserved in merged mesh")
        print("This is expected behavior for some PyVista boolean operations")
        
        # Fallback: Use spatial identification
        print("\nFalling back to spatial identification...")
        spatial_results = identify_materials_spatially(merged_mesh, original_meshes)
        return False, spatial_results


def test_sphere_merge_identification():
    """Main test function - clearly separated concerns."""
    
    # 1. SPHERE GENERATION
    sphere1, sphere2 = create_test_spheres()
    
    # 2. MERGE OPERATION  
    merged_sphere = perform_merge_operation(sphere1, sphere2)
    
    # 3. MATERIAL ANALYSIS
    preserved, analysis_result = analyze_material_preservation(merged_sphere, [sphere1, sphere2])
    
    # 4. VISUALIZATION
    print("\nCreating PyVista visualization...")
    create_visualization(sphere1, sphere2, merged_sphere)
    
    return merged_sphere


def check_material_preservation(merged_mesh):
    """Check if material information is preserved in merged mesh (transferable function)."""
    if 'material_id' not in merged_mesh.cell_data:
        return False, "Material IDs not found in merged mesh"
    
    material_ids = merged_mesh.cell_data['material_id']
    unique_materials = np.unique(material_ids)
    
    results = {
        'preserved': True,
        'unique_materials': unique_materials,
        'material_stats': {}
    }
    
    print(f"Unique material IDs found: {unique_materials}")
    
    for mat_id in unique_materials:
        count = np.sum(material_ids == mat_id)
        percentage = (count / len(material_ids)) * 100
        results['material_stats'][mat_id] = {'count': count, 'percentage': percentage}
        print(f"  Material {mat_id}: {count} cells ({percentage:.1f}%)")
    
    return True, results


def extract_regions_by_material(merged_mesh):
    """Extract regions from merged mesh by material ID (transferable function)."""
    if 'material_id' not in merged_mesh.cell_data:
        return {}
    
    material_ids = merged_mesh.cell_data['material_id']
    unique_materials = np.unique(material_ids)
    regions = {}
    
    print("Extracting regions by material...")
    
    for mat_id in unique_materials:
        # Create mask for this material
        mask = material_ids == mat_id
        cell_indices = np.where(mask)[0]
        
        # Extract cells belonging to this material
        region = merged_mesh.extract_cells(cell_indices)
        regions[mat_id] = region
        
        print(f"  Region {mat_id}: {region.n_cells} cells, {region.n_points} points")
        
        # Verify the material ID is consistent
        if 'material_id' in region.cell_data:
            region_materials = np.unique(region.cell_data['material_id'])
            if len(region_materials) == 1 and region_materials[0] == mat_id:
                print(f"    ✓ Material identification successful")
            else:
                print(f"    ✗ Material identification failed: {region_materials}")
    
    return regions


def identify_materials_spatially(merged_mesh, original_meshes):
    """Identify regions by spatial proximity to original meshes (transferable function)."""
    from scipy.spatial import cKDTree
    
    print("Building spatial index for identification...")
    
    # Get centroids of merged mesh cells
    merged_centroids = merged_mesh.cell_centers().points
    
    # Build KD-trees for original meshes and find closest matches
    distances = []
    for i, original_mesh in enumerate(original_meshes):
        orig_centroids = original_mesh.cell_centers().points
        tree = cKDTree(orig_centroids)
        dist, _ = tree.query(merged_centroids)
        distances.append(dist)
    
    # Assign to closest original mesh (1-indexed material IDs)
    distances = np.array(distances)
    material_ids = np.argmin(distances, axis=0) + 1
    
    # Add identified materials to merged mesh
    merged_mesh.cell_data['identified_material'] = material_ids
    
    # Report results
    unique_materials = np.unique(material_ids)
    print(f"Spatial identification found {len(unique_materials)} materials:")
    
    results = {'unique_materials': unique_materials, 'material_stats': {}}
    
    for mat_id in unique_materials:
        count = np.sum(material_ids == mat_id)
        percentage = (count / len(material_ids)) * 100
        results['material_stats'][mat_id] = {'count': count, 'percentage': percentage}
        print(f"  Material {mat_id}: {count} cells ({percentage:.1f}%)")
    
    return results


def create_visualization(sphere1, sphere2, merged_sphere):
    """Create and display PyVista visualization of the spheres before and after merging."""
    
    # Create plotter with subplots
    plotter = pv.Plotter(shape=(1, 3), window_size=[1200, 400])
    
    # Plot original spheres
    plotter.subplot(0, 0)
    plotter.add_mesh(sphere1, color='red', opacity=0.7, label='Sphere 1')
    plotter.add_mesh(sphere2, color='blue', opacity=0.7, label='Sphere 2')
    plotter.add_title("Original Spheres")
    plotter.add_legend()
    
    # Plot merged result
    plotter.subplot(0, 2)
    
    # Color by material if available
    if 'material_id' in merged_sphere.cell_data:
        plotter.add_mesh(merged_sphere, scalars='material_id', 
                        cmap=['red', 'blue'], show_scalar_bar=True)
        plotter.add_title("Merged (Colored by Material)")
    elif 'identified_material' in merged_sphere.cell_data:
        plotter.add_mesh(merged_sphere, scalars='identified_material', 
                        cmap=['red', 'blue'], show_scalar_bar=True)
        plotter.add_title("Merged (Spatially Identified)")
    else:
        plotter.add_mesh(merged_sphere, color='lightgray')
        plotter.add_title("Merged Sphere")
    
    # Show the visualization
    plotter.show()
    return plotter


if __name__ == "__main__":
    print("Testing sphere merge and identification workflow...")
    print("=" * 50)
    
    # Run the test
    merged = test_sphere_merge_identification()
    
    print("\n" + "=" * 50)
    print("Test completed!")
