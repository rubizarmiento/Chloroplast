import trimesh
import pyvista as pv
import numpy as np
import os

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
    
def write_mtl_and_update_obj(obj_filename, mtl_filename, material_name, color_rgb, transparency=1.0, shininess=10):
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