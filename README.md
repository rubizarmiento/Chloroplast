# Chloroplast

## Running the `thylakoid_mesh.ipynb` Notebook locally
This step is preffered as it allows to visualize 3D models interactively in Jupyter.


Follow these steps to set up your development environment:

### 1. Duplicate repository
```bash
git clone https://github.com/rubizarmiento/Chloroplast.git
cd Chloroplast
```

### 2. Create a Python Environment
```bash
python3.10 -m venv "thylakoid_mesh"
```

### 3. Activate the Python Environment
```bash
source thylakoid_mesh/bin/activate
```

### 4. Install Jupyter
```bash
pip install --upgrade pip
pip install jupyter
```

### 5. Create a Jupyter Kernel
```bash
python -m ipykernel install --user --name=env --display-name "thylakoid_mesh"
```

### 6. Install Required Packages
```bash
pip install -r requirements.txt
``` 

### 7. Launch Jupyter Notebook `thylakoid_mesh.ipynb`
```bash
jupyter-lab thylakoid_mesh.ipynb
```
or in VS CODE
```bash
code thylakoid_mesh.ipynb
```




# Project Description

## Background
The thylakoid system is the main structure inside a chloroplast and is where the photosynthetic proteins are located.

## Objective
`Generate a 3D mesh that can be used to place proteins and lipids on membrane surfaces.`

## Challenges

1. **Incomplete reference data:** Only a partial cryo-EM structure is available, but a geometric description of the system exists.
2. **Periodic model required:** A periodic model enables placement in a periodic simulation box and exploitation of structural symmetry.
3. **Homogeneous face distribution:** Proteins must be distributed uniformly, requiring an even distribution of mesh faces.

## How the Challenges Are Addressed

1. Based on cryo-EM data and associated reported statistics, ad-hoc geometries are generated using experimentally derived parameters.
2. Symmetry operations are used to assemble the system; a representative section is then used as the periodic unit.
3. All geometries can be generated with a user-defined `target_resolution`, producing meshes with uniformly sized faces.

---

**`Goal of this notebook:`** 
Showcase the thylakoid membrane system and demonstrate the effect of `target_resolution` on mesh quality

## Project Structure

- `thylakoid_mesh.ipynb` - Main Jupyter notebook for mesh processing
- `blender_scripts/` - Scripts for Blender integration
- `Reference_objs/` - Reference 3D objects and models
- `tests/` - Unit tests for the project
- `requirements.txt` - Python dependencies





