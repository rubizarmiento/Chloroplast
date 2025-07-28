# Chloroplast

A Python project for the visualization of a chloroplast using PyVista.

## Setting up the Python Environment and Jupyter Kernel

Follow these steps to set up your development environment:

### 1. Create a Python Environment
```bash
python3.10 -m venv "env"
```

### 2. Activate the Python Environment
```bash
source env/bin/activate
```

### 3. Install Jupyter
```bash
pip install --upgrade pip
pip install jupyter
```

### 4. Create a Jupyter Kernel
```bash
python -m ipykernel install --user --name=env --display-name "default"
```

### 5. Install Required Packages
```bash
pip install -r requirements.txt
```

## Project Structure

- `remeshing.ipynb` - Main Jupyter notebook for mesh processing
- `blender_scripts/` - Scripts for Blender integration
- `Reference_objs/` - Reference 3D objects and models
- `tests/` - Unit tests for the project
- `requirements.txt` - Python dependencies

## Usage

1. Follow the setup instructions above
2. Launch Jupyter Lab or Jupyter Notebook
3. Open `remeshing.ipynb` to begin working with the project




