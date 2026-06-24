import numpy as np
import pyacvd
import yaml

from .pyvista_geometries import cylinder_with_concentric_circles, generate_helix_spline, create_rectangular_tube, create_elliptical_helix
from .simple_math_geometry import area_cylinder, helix_pitch_from_tilt, helix_radius_from_tilt, calculate_helix_length
from .mesh_utils import mesh_resolution

def read_model_parameters(params_path="model_parameters.yaml"):
    """
    Reads the model parameters from a YAML file and returns them as a dictionary.

    Parameters
    ----------
    params_path : str
        Path to the YAML file containing model parameters.

        -Expected parameters in the YAML file:
        radius_granum : float
            Radius of the granum cylinder in nm.
        height_crosssection : float
            Height of the granum cylinder in nm.
        membrane_thickness : float
            Membrane thickness in nm.
        stromal_gap : float
            Gap between stacked grana in nm.
        n_granum : int
            Number of grana disks in the stack.
        tilt_angle : float
            Tilt angle of the right-handed helix in degrees.
        inner_pore : float
            Inner radius of the left helix cross-section in nm.

    Returns
    -------
    dict
        Dictionary containing the model parameters.
    """
    with open(params_path) as f:
        params = yaml.safe_load(f)
    # Assert that all required parameters are present
    type_assertions = {
        "radius_granum": (int, float),
        "height_crosssection": (int, float),
        "membrane_thickness": (int, float),
        "stromal_gap": (int, float),
        "n_granum": int,
        "tilt_angle": (int, float),
        "inner_pore": (int, float),
    }
    for param in type_assertions:
        if param not in params:
            raise ValueError(f"Missing required parameter: {param}")
        if not isinstance(params[param], type_assertions[param]):
            raise TypeError(f"Parameter '{param}' must be of type {type_assertions[param]}, got {type(params[param])} instead.")
    return dict(params)

def get_model_height(dict_params):
    """
    Calculates the total height of the granum stack.

    Parameters
    ----------
    dict_params : dict
        Dictionary containing the model parameters.

        -The following parameters are expected in the dictionary:
        
        height_crosssection : float
            Height of the granum cylinder in nm.
        membrane_thickness : float
            Membrane thickness in nm.
        stromal_gap : float
            Gap between stacked grana in nm.
        n_granum : int
            Number of grana disks in the stack.

    Returns
    -------
    float
        Total height in nm
    """
    height_crosssection=dict_params['height_crosssection']
    membrane_thickness=dict_params['membrane_thickness']
    stromal_gap=dict_params['stromal_gap']
    n_granum=dict_params['n_granum']

    granum_separation   = height_crosssection + membrane_thickness + stromal_gap
    total_height_stacks = (
        n_granum * ((granum_separation / 2) + height_crosssection) - granum_separation / 2
    )

    return total_height_stacks

def get_right_helix_parameters(dict_params):
    """
    Calculate the parameters for building a right-handed helix mesh.

    Parameters
    ----------
    dict_params : dict
        Dictionary containing the model parameters.

        -The following parameters are expected in the dictionary:
        
        tilt_angle : float
            Tilt angle of the right-handed helix in degrees.
        radius_granum : float
            Radius of the granum cylinder in nm.
        height_crosssection : float
            Height of the granum cylinder in nm.
    
    Returns
    -------
    dict
        Dictionary containing the calculated parameters for the right-handed helix mesh.
    """
    p = dict_params
    radius_granum = p['radius_granum']
    height_crosssection = p['height_crosssection']
    tilt_angle = p['tilt_angle']

    radius_spline_right    = radius_granum + height_crosssection
    width_right_helix = (height_crosssection * 2) + 1   # 1 nm forced intersection with granum
    model_height = get_model_height(dict_params=dict_params)
    
    pitch_right         = helix_pitch_from_tilt(radius_spline_right, tilt_angle)
    n_turns_right_helix = model_height / pitch_right
    spline_length       = calculate_helix_length(
        radius=radius_spline_right, pitch=pitch_right, turns=n_turns_right_helix
    )
    print(f"Radius of right helix spline: {radius_spline_right:.2f} nm")
    print(f"Width of right helix: {width_right_helix:.2f} nm")  
    total_radius_right_helix = radius_granum + width_right_helix

    total_height_helix = pitch_right * n_turns_right_helix

    return dict(
        height_crosssection = height_crosssection,
        radius_spline       = radius_spline_right,
        width_helix         = width_right_helix,
        total_radius_helix  = total_radius_right_helix,
        pitch               = pitch_right,
        n_turns             = n_turns_right_helix,
        spline_length       = spline_length,
        total_height_helix  = total_height_helix,
    )

def get_left_helix_parameters(dict_params):
    """
    Calculate the parameters for building a left-handed helix mesh.

    Parameters
    ----------
    dict_params : dict
        Dictionary containing the model parameters.

        -The following parameters are expected in the dictionary:

        radius_granum : float
            Radius of the granum cylinder in nm.
        height_crosssection : float
            Height of the granum cylinder in nm.
        tilt_angle : float
            Tilt angle shared with the right-handed helix in degrees.
        membrane_thickness : float
            Membrane thickness in nm.
        stromal_gap : float
            Gap between stacked grana in nm.
        n_granum : int
            Number of grana disks in the stack.
        inner_pore : float
            Inner radius of the left helix cross-section in nm.

    Returns
    -------
    dict
        Dictionary containing the calculated parameters for the left-handed helix mesh.
    """
    p = dict_params
    radius_granum       = p['radius_granum']
    height_crosssection = p['height_crosssection']
    tilt_angle    = p['tilt_angle']
    inner_pore          = p['inner_pore']

    # Right helix geometry — needed for pitch and total helix height
    radius_spline_right      = radius_granum + height_crosssection
    pitch_right              = helix_pitch_from_tilt(radius_spline_right, tilt_angle)
    model_height             = get_model_height(dict_params=dict_params)
    n_turns_right            = model_height / pitch_right
    total_height_right_helix = pitch_right * n_turns_right

    # Left helix geometry
    pitch_left              = pitch_right / 4
    radius_spline_left      = helix_radius_from_tilt(pitch_left, tilt_angle)
    width_left_helix        = (radius_spline_left - inner_pore) * 2
    total_radius_left_helix = radius_spline_left + width_left_helix / 5
    width_right_helix       = (height_crosssection * 2) + 1
    d_granum_helix          = radius_granum + total_radius_left_helix + width_right_helix * 0.75

    n_turns_left   = total_height_right_helix / pitch_left
    spline_length  = calculate_helix_length(
        radius=radius_spline_left, pitch=pitch_left, turns=n_turns_left
    )

    print(f"Radius of left helix spline: {radius_spline_left:.2f} nm")
    print(f"Width of left helix: {width_left_helix:.2f} nm")
    print(f"d_granum_helix: {d_granum_helix:.2f} nm")

    return dict(
        height_crosssection = height_crosssection,
        radius_spline       = radius_spline_left,
        width_helix         = width_left_helix,
        total_radius_helix  = total_radius_left_helix,
        pitch               = pitch_left,
        n_turns             = n_turns_left,
        spline_length       = spline_length,
        d_granum_helix      = d_granum_helix,
    )


def build_right_helix_thylakoid_mesh(dict_params, target_resolution=2, verbose=True):
    """
    Build a right-handed helix mesh (rectangular tube along a helical spline) for the thylakoid.

    Parameters
    ----------
    target_resolution : float
        Target area per triangle face in nm^2. Lower = finer mesh.
    dict_params : dict
        Dictionary containing the model parameters.
        
        -The following parameters are expected in the dictionary:
        radius_granum : float
            Radius of the granum cylinder in nm.
        height_crosssection : float
            Height of the granum cylinder in nm.
        tilt_angle : float
            Tilt angle of the right-handed helix in degrees.

        -The following parameters are derived with get_right_helix_parameters():
        radius_spline : float
            Radius of the helix spline in nm.
        width_helix : float
            Width of the rectangular tube cross-section in nm.
        pitch : float
            Axial rise per full turn of the helix in nm.
        n_turns : float
            Number of turns spanning the full grana stack.
        spline_length : float
            Length of the helix spline in nm.
        
    Returns
    -------
    tube_mesh : pyvista.PolyData
        Rectangular tube mesh starting at z=0.
    """
    radius_granum = dict_params['radius_granum']
    height_crosssection = dict_params['height_crosssection']
    tilt_angle = dict_params['tilt_angle']
    
    params = get_right_helix_parameters(dict_params=dict_params)
    radius_spline = params['radius_spline']
    width_helix = params['width_helix']
    pitch = params['pitch']
    n_turns = params['n_turns']
    spline_length = params['spline_length']

    return build_helix_mesh(
        height_crosssection=height_crosssection,
        radius_spline=radius_spline,
        width_helix=width_helix,
        pitch=pitch,
        n_turns=n_turns,
        spline_length=spline_length,
        chirality='right',
        target_resolution=target_resolution,
        verbose=verbose
    )

def build_left_helix_thylakoid_mesh(dict_params, target_resolution=2, verbose=True):
    """
    Build a left-handed helix mesh (rectangular tube along an elliptical helical spline).

    The mesh is placed at d_granum_helix distance from the origin, ready for
    4-fold rotational copies via build_rotational_array.

    Parameters
    ----------
    dict_params : dict
        Dictionary containing the model parameters (output of read_model_parameters).

        -The following parameters are derived with get_left_helix_parameters():
        height_crosssection : float
            Height of the tube cross-section in nm.
        radius_spline : float
            Radius of the left helix spline in nm.
        width_helix : float
            Width of the rectangular tube cross-section in nm.
        pitch : float
            Axial rise per full turn in nm.
        n_turns : float
            Number of turns spanning the full grana stack.
        spline_length : float
            Arc length of the helix in nm.
        d_granum_helix : float
            Radial distance from origin to the helix center (nm).

    target_resolution : float
        Target area per triangle face in nm^2. Lower = finer mesh.
    verbose : bool
        Print mesh statistics when True.

    Returns
    -------
    tube_mesh : pyvista.PolyData
        Rectangular tube mesh positioned at d_granum_helix from the origin.
    """
    params = get_left_helix_parameters(dict_params=dict_params)
    height_crosssection = params['height_crosssection']
    radius_spline       = params['radius_spline']
    width_helix         = params['width_helix']
    pitch               = params['pitch']
    n_turns             = params['n_turns']
    spline_length       = params['spline_length']
    d_granum_helix      = params['d_granum_helix']

    target_side       = target_resolution ** 0.5
    n_points_spline   = int(spline_length / (2 * target_side)) + 1
    resolution_width  = int(width_helix         / target_side) + 1
    resolution_height = int(round(height_crosssection / target_side, 0)) + 1

    helix_points = create_elliptical_helix(
        x0=0, y0=-d_granum_helix, z0=height_crosssection,
        radius_x=radius_spline,
        radius_y=radius_spline,
        pitch=pitch,
        turns=n_turns,
        n_points=n_points_spline,
        chirality='left',
    )

    tube_mesh = create_rectangular_tube(
        helix_points,
        width=width_helix,
        height=height_crosssection,
        height_resolution=resolution_height,
        width_resolution=resolution_width,
        closed=False,
    )

    if verbose:
        mesh_resolution(tube_mesh)

    return tube_mesh


def build_granum_mesh(
    radius_granum=125,
    height_crosssection=18,
    target_resolution=2,
    membrane_thickness=4,
    stromal_gap=3,
    center=[0, 0, 0],
    direction=[0, 0, 1],
    verbose=True,
):
    """
    Build a single remeshed granum disk (thylakoid membrane cylinder).

    Parameters
    ----------
    radius_granum : float
        Radius of the granum cylinder in nm.
    height_crosssection : float
        Height of the granum cylinder in nm.
    target_resolution : float
        Target area per triangle face in nm^2. Lower = finer mesh.
    membrane_thickness : float
        Membrane thickness in nm (used only for info output).
    stromal_gap : float
        Gap between stacked grana in nm (used only for info output).
    center : array_like, optional
        Center of the cylinder. Defaults to [0, 0, 0].
    direction : array_like, optional
        Axis direction of the cylinder. Defaults to [0, 0, 1].
    verbose : bool
        Print mesh statistics when True.

    Returns
    -------
    remesh : pyvista.PolyData
        Remeshed granum cylinder with approximately target_resolution nm^2 per face.
    """
    # Derived internal resolution parameters — high enough to feed the remesher
    radial_resolution   = 1080
    height_resolution   = int(height_crosssection * 10)
    concentric_circles  = int(radius_granum * 10)

    granum_area    = area_cylinder(radius=radius_granum, height=height_crosssection)
    target_n_faces = int(granum_area / target_resolution)

    if verbose:
        granum_separation = height_crosssection + membrane_thickness + stromal_gap
        print(f"Area of a single granum: {granum_area:.0f} nm^2")
        print(f"Target faces for {target_resolution} nm^2/face: {target_n_faces}")
        print(f"Granum separation (with stromal gap): {granum_separation} nm")

    # Build high-resolution source mesh
    cylinder_mesh = cylinder_with_concentric_circles(
        radius=radius_granum,
        height=height_crosssection,
        radial_resolution=radial_resolution,
        height_resolution=height_resolution,
        concentric_circles=concentric_circles,
        center=center,
        direction=direction,
    ).triangulate()

    # Remesh to target resolution via ACVD clustering
    clus = pyacvd.Clustering(cylinder_mesh)
    clus.cluster(int(target_n_faces / 2))
    remesh = clus.create_mesh().clean()

    if verbose:
        mesh_resolution(remesh)

    return remesh

def build_helix_mesh(
    height_crosssection,
    radius_spline,
    width_helix,
    pitch,
    n_turns,
    spline_length,
    chirality='right',
    target_resolution=2,
    verbose=True,
):
    """
    Build a helix mesh (rectangular tube along a helical spline).

    Intended to be called with the output of a calculate_*_parameters()
    function unpacked via **: build_helix_mesh(**params, chirality='right')

    Parameters
    ----------
    height_crosssection : float
        Height of the tube cross-section (nm)
    radius_spline : float
        Radius of the helix spline (nm).
    width_helix : float
        Width of the rectangular tube cross-section (nm).
    pitch : float
        Axial rise per full turn of the helix (nm).
    n_turns : float
        Number of turns spanning the full grana stack.
    spline_length : float
        Arc length of the helix (nm). Used to set spline segment count.
    chirality : str
        'right' or 'left'.
    target_resolution : float
        Target area per triangle face in nm^2. 
    verbose : bool
        Print mesh statistics when True.

    Returns
    -------
    tube_mesh : pyvista.PolyData
        Rectangular tube mesh starting at z=0.
    """
    target_side = (target_resolution ** 0.5) # target side length of a square face

    n_points_spline   = int(spline_length     / (2 * target_side)) + 1 # The 2 factor takes into account the triangulation
    resolution_width  = int(width_helix       /      target_side ) + 1
    resolution_height = int(round(height_crosssection / target_side, 0)) + 1

    helix_points = generate_helix_spline(
        x0=0, y0=0, z0=0,
        radius=radius_spline,
        pitch=pitch,
        turns=n_turns,
        n_points=n_points_spline,
        chirality=chirality,
    )

    tube_mesh = create_rectangular_tube(
        helix_points,
        width=width_helix,
        height=height_crosssection,
        height_resolution=resolution_height,
        width_resolution=resolution_width,
        closed=False,
    )

    # Center in x/y only — z stays at 0 to align with grana stacks
    center    = np.array(tube_mesh.center)
    center[2] = 0.0
    tube_mesh.points -= center

    if verbose:
        mesh_resolution(tube_mesh)

    return tube_mesh


def build_left_helix_mesh(
    radius_granum=125,
    height_crosssection=18,
    membrane_thickness=4,
    stromal_gap=3,
    n_granum=10,
    tilt_angle=18.45,
    inner_pore=25,
    target_resolution=2,
    verbose=True,
):
    """
    Build a left-handed helix mesh (rectangular tube along an elliptical helical spline).

    The left helix pitch is 1/4 of the right helix pitch and its tilt angle matches
    the right helix. The mesh is placed at d_granum_helix distance from the granum
    center (not centered at the origin), ready for 4-fold rotational copies.

    Parameters
    ----------
    radius_granum : float
        Radius of the granum cylinder in nm.
    height_crosssection : float
        Height of the granum cylinder in nm.
    membrane_thickness : float
        Membrane thickness in nm.
    stromal_gap : float
        Gap between stacked grana in nm.
    n_granum : int
        Number of grana disks in the stack.
    tilt_angle : float
        Tilt angle of the right-handed helix in degrees (left helix shares this value).
    inner_pore : float
        Inner radius of the left helix cross-section in nm.
    target_resolution : float
        Target area per triangle face in nm^2. Lower = finer mesh.
    verbose : bool
        Print mesh statistics when True.

    Returns
    -------
    tube_mesh_left : pyvista.PolyData
        Rectangular tube mesh positioned at d_granum_helix from the origin.
    """
    # --- Derive right helix geometry (needed for pitch and total height)
    radius_spline_right      = radius_granum + height_crosssection
    pitch_right         = helix_pitch_from_tilt(radius_spline_right, tilt_angle)
    granum_separation   = height_crosssection + membrane_thickness + stromal_gap
    total_height_stacks = n_granum * ((granum_separation / 2) + height_crosssection) - granum_separation / 2
    n_turns_right_helix = total_height_stacks / pitch_right
    height_spline_right = (pitch_right * n_turns_right_helix) - height_crosssection
    total_height_right_helix = height_spline_right + height_crosssection

    # --- Left helix geometry
    pitch_left          = pitch_right / 4
    tilt_angle_left     = tilt_angle
    radius_spline_left       = helix_radius_from_tilt(pitch_left, tilt_angle_left)
    width_left_helix    = (radius_spline_left - inner_pore) * 2
    total_radius_left_helix = radius_spline_left + width_left_helix / 5

    # Hardcoded: right helix overlaps granum by 1 nm (mirrors build_right_helix_mesh)
    width_right_helix   = (height_crosssection * 2) + 1
    d_granum_helix      = radius_granum + total_radius_left_helix + width_right_helix * 0.75

    n_turns_left_helix  = total_height_right_helix / pitch_left
    spline_length_left  = calculate_helix_length(
        radius=radius_spline_left, pitch=pitch_left, turns=n_turns_left_helix
    )

    resolution_spline  = int(spline_length_left / target_resolution) + 1
    resolution_width   = int(width_left_helix / (2 * target_resolution)) + 1
    resolution_height  = int(round(height_crosssection / (2 * target_resolution), 0)) + 1

    if verbose:
        rz_spline = (pitch_left * n_turns_left_helix) - height_crosssection
        print(f"Total height of right helix (reference): {total_height_right_helix:.2f} nm")
        print(f"Turns: {n_turns_left_helix:.2f}  |  Pitch: {pitch_left:.2f} nm")
        print(f"Spline radius: {radius_spline_left:.2f} nm  |  Total left helix radius: {total_radius_left_helix:.2f} nm")
        print(f"Width of left helix: {width_left_helix:.2f} nm  |  Spline z-height: {rz_spline:.2f} nm")
        print(f"d_granum_helix: {d_granum_helix:.2f} nm")
        print(f"Spline length: {spline_length_left:.2f} nm  →  {resolution_spline} segments")

    helix_points = create_elliptical_helix(
        x0=0, y0=-d_granum_helix, z0=0,
        radius_x=radius_spline_left,
        radius_y=radius_spline_left,
        pitch=pitch_left,
        turns=n_turns_left_helix,
        n_points=resolution_spline,
        chirality='left',
    )

    tube_mesh_left = create_rectangular_tube(
        helix_points,
        width=width_left_helix,
        height=height_crosssection,
        height_resolution=resolution_height,
        width_resolution=resolution_width,
        closed=False,
    )

    if verbose:
        mesh_resolution(tube_mesh_left)

    return tube_mesh_left
