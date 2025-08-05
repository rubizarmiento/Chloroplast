#!/usr/bin/env python3
"""
Author: Rubi Zarmiento-Garcia
Date: 05/08/2025
Merge, differentiate or intersect multiple 3D models using PyVista boolean operations.
Assign IDs to each model to identify them before and after performing boolean operations.
Formats can be .obj, .vtk, .vtp, or .ply. or any other format supported by PyVista.

Arguments:
    -type --type: Boolean operation type (union, intersection, difference)
    -f, --files: List of input model files to merge (e.g. model1.obj model2.obj). Accepted formats are .obj, .vtk, .vtp, or .ply. or any other format supported by PyVista.
    -id, --ids: List of IDs to identify the models before and after merging (e.g. 1 2 3)
    -o, --output: Output file for the model. Format can be .obj, .vtk, .vtp, or .ply. or any other format supported by PyVista.
Usage: python test_merge_helices.py -type union -f model1.obj model2.obj model3.obj -o merged.obj
"""

import pyvista as pv
import argparse
import sys
from pathlib import Path
import json
import numpy as np

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Merge, differentiate or intersect multiple 3D models using PyVista. Assign IDs to each model to identify them before and after performing boolean operations.")
    parser.add_argument('-type', '--type', choices=['union', 'intersection', 'difference'], default='union', help='Type of boolean operation to perform (default: union)')
    parser.add_argument('-f', '--files', nargs='+', required=True, help='Input model files to merge. Accepted formats are .obj, .vtk, .vtp, or .ply. or any other format supported by PyVista.')
    parser.add_argument('-ids', '--ids', nargs='+', required=True, help='IDs to identify the models before and after merging, e.g. 1 2 3')
    parser.add_argument('-o', '--output', required=True, help='Output file for the model')
    args = parser.parse_args()
    return args

def load_models(file_paths,ids):
    """Load models from file paths."""
    models = []
    for file_path in file_paths:
        try:
            model = pv.read(file_path)
            models.append(model)
            model = add_id(model, ids.pop(0))  # Add ID to the model, .pop removes the first ID from the list
            print(f"Loaded: {file_path} ({model.n_points} pts, {model.n_cells} cells with ID {model.cell_data['id'][0]})")
        except Exception as e:
            print(f"ERROR loading {file_path}: {e}")
            sys.exit(1)
    return models

def add_id(mesh, id):
    """
    Add id label to the cell data of the mesh.

    Args:
        mesh (pyvista.PolyData): The mesh to which the id will be added.
        id (int): The id to add to the mesh.
    Returns:
        pyvista.PolyData: The mesh with the id added to its cell data.
    """

    # Add material ID as cell data (standard VTK approach)
    mesh.cell_data['id'] = np.full(mesh.n_cells, id)
    return mesh

def boolean_models(models,type='union'):
    """Modify models using progressive boolean union."""
    if len(models) < 2:
        print("Need at least 2 models to perform a boolean operation")
        return models[0] if models else None
    
    result = models[0]
    print(f"Starting with model 1: {result.n_points} points")
    
    for i, model in enumerate(models[1:], 2):
        try:
            print(f"Merging model {i}...")
            if type == 'union':
                result = result.boolean_union(model)
            elif type == 'intersection':
                result = result.boolean_intersection(model)
            elif type == 'difference':
                result = result.boolean_difference(model)
            print(f"  Result: {result.n_points} points, {result.n_cells} cells")
        except Exception as e:
            print(f"ERROR merging model {i} with type {type}: {e}")
            print("Maybe the models are too complex for PyVista :(")
            print("You could try cleaning or triangulating your system with Pyvista beforehand.")
            if type == 'union':
                print("Union may fail if models are not manifold or have complex geometry.")
            elif type == 'intersection':
                print("Intersection may fail if models do not overlap.")
            elif type == 'difference':
                print("Difference may fail if models do not have a common region.")
            sys.exit(1)
            break
    
    return result

def save_ids(mesh, output_path):
    """Save OBJ + IDs as separate JSON file."""    
    if 'id' in mesh.cell_data:
        id_path = output_path.replace('.obj', '_ids.json')
        with open(id_path, 'w') as f:
            json.dump(mesh.cell_data['id'].tolist(), f)
        
def main():

    args = parse_args()
    
    # Validate input files
    for file_path in args.files:
        if not Path(file_path).exists():
            print(f"ERROR: File not found: {file_path}")
            sys.exit(1)
    
    print(f"Merging {len(args.files)} models...")
    
    # Load models
    models = load_models(args.files, args.ids)
    
    # Merge models
    modified = boolean_models(models, type='union')  # Default to union operation

    if modified is None:
        print("ERROR: Boolean operation failed")
        print("Maybe the models are too complex for PyVista :(")
        print("You could try cleaning or triangulating your system with Pyvista beforehand.")
        sys.exit(1)
    
    # Save result
    try:
        modified.save(args.output)
        print(f"Saved model to: {args.output}")
        print(f"Final mesh: {modified.n_points} points, {modified.n_cells} cells")
        save_ids(modified, args.output)
        print(f"\nSaved IDs to: {args.output.replace('.obj', '_ids.json')}")
        print(f"They can be loaded as:\n import pyvista as pv\n mesh = pv.read('{args.output}')\n with open('{args.output.replace('.obj', '_ids.json')}', 'r') as f:\n    ids = json.load(f)\n mesh.cell_data['id'] = ids")
        print(f"\nAlternatively, save the mesh as .vtk")
    except Exception as e:
        print(f"ERROR saving to {args.output}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

