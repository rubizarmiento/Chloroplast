import numpy as np
import pyvista as pv
from martini_mesh.mesh_utils import merge_meshes

def build_rotational_array(mesh, n_copies=4, theta_0=0.0, point=None):
    """
    Create N evenly-spaced rotational copies of a mesh around the z-axis and
    combine them into a single mesh.

    Parameters
    ----------
    mesh : pyvista.PolyData
        The mesh to copy.
    n_copies : int
        Number of copies. Copies are spaced 360 / n_copies degrees apart.
    theta_0 : float
        Rotation angle in degrees for the first copy.
    point : array_like, optional
        [x, y, z] center of rotation. Defaults to the origin [0, 0, 0].

    Returns
    -------
    copies : list of pyvista.PolyData
        Individual rotated copies in order from theta_0.
    """
    if point is None:
        point = [0.0, 0.0, 0.0]

    step = 360.0 / n_copies
    copies = []
    for i in range(n_copies):
        rotated = mesh.copy()
        rotated.rotate_z(theta_0 + i * step, point=point, inplace=True)
        copies.append(rotated)
    
    copies = merge_meshes(copies)  # Combine into a single mesh

    return copies


def build_connector_up_down(mesh, pitch_left, height_crosssection, z_margin=4, z_offset=-1):
    """
    Create the full periodic connector array: 4 copies at the lower level (down)
    and 4 copies at the upper level (flipped + translated up).

    Parameters
    ----------
    mesh : pyvista.PolyData
        Single connector mesh (output of build_connector_mesh).
    pitch_left : float
        Pitch of the left-handed helix in nm. Sets the z offset of the upper level.
    height_crosssection : float
        Cross-section height in nm (= height_granum). Part of the upper z offset.
    z_margin : float
        Extra z gap added to the upper level offset in nm. Default 4.
    z_offset : float
        Z offset for the lower level connectors in nm. Default -1.

    Returns
    -------
    pyvista.PolyData
        All 8 connector copies (4 down + 4 up) merged into a single mesh.
    """
    connectors_down = build_rotational_array(mesh, n_copies=4)
    connectors_down.points[:, 2] -= z_offset

    up_template = mesh.copy()
    up_template.rotate_x(180, inplace=True)
    up_template.points[:, 2] += 0.5 * pitch_left + height_crosssection + z_margin
    connectors_up = build_rotational_array(up_template, n_copies=4)
    connectors_up.points[:, 2] += height_crosssection + z_offset

    return connectors_down + connectors_up


def build_periodic_connectors(connectors, pitch_left, total_height_right_helix):
    """
    Replicate a connector unit (up + down) periodically along z to span the
    full height of the right-handed helix.

    Parameters
    ----------
    connectors : pyvista.PolyData
        Single connector unit containing both up and down copies
        (output of build_connector_up_down).
    pitch_left : float
        Step size between connector levels in nm (= pitch of left helix).
    total_height_right_helix : float
        Total z extent of the right helix in nm. Copies are placed within
        [-total_height_right_helix/2, +total_height_right_helix/2].

    Returns
    -------
    pyvista.PolyData
        All periodic connector copies merged into a single mesh.
    """
    z0 = connectors.center[2]
    n  = max(1, round(total_height_right_helix / pitch_left))

    # Helix spans [0, total_height_right_helix] — step upward from z0
    z_centers = z0 + np.arange(n) * pitch_left

    copies = []
    for z_center in z_centers:
        copy = connectors.copy()
        copy.points[:, 2] += z_center - z0
        copies.append(copy)

    print(f"z0: {z0:.2f} nm  |  {n} levels at: {np.round(z_centers, 1)}")
    return merge_meshes(copies)


def build_connector_thylakoid_array(dict_params, connector_mesh):
    """
    Assemble the full periodic connector array from a single connector mesh.

    Wraps build_connector_up_down and build_periodic_connectors using
    dict_params for all geometry values.

    Parameters
    ----------
    dict_params : dict
        Output of read_model_parameters().
    connector_mesh : pyvista.PolyData
        Single connector patch (output of build_connector_thylakoid_mesh or
        build_connector_from_loop).

    Returns
    -------
    pyvista.PolyData
        All periodic connector copies merged into a single mesh.
    """
    import io, contextlib
    from .builders import get_right_helix_parameters, get_left_helix_parameters

    with contextlib.redirect_stdout(io.StringIO()):
        right_params = get_right_helix_parameters(dict_params)
        left_params  = get_left_helix_parameters(dict_params)

    connectors = build_connector_up_down(
        connector_mesh,
        pitch_left=left_params['pitch'],
        height_crosssection=dict_params['height_crosssection'],
    )

    return build_periodic_connectors(
        connectors,
        pitch_left=left_params['pitch'],
        total_height_right_helix=right_params['total_height_helix'],
    )


def build_linear_array(mesh, n_copies=10, separation=25, start=None, direction=None):
    """
    Create N copies of a mesh stacked along a direction vector.

    Parameters
    ----------
    mesh : pyvista.PolyData
        The mesh to copy.
    n_copies : int
        Number of copies (including the first at `start`).
    separation : float
        Distance between consecutive copies in nm.
    start : array_like, optional
        [x, y, z] position of the first copy. Defaults to [0, 0, 0].
    direction : array_like, optional
        Unit vector defining the stacking axis. Defaults to [0, 0, 1].

    Returns
    -------
    copies : list of pyvista.PolyData
        Copies positioned along the stacking axis.
    """
    if start is None:
        start = np.array([0.0, 0.0, 0.0])
    else:
        start = np.asarray(start, dtype=float)

    if direction is None:
        direction = np.array([0.0, 0.0, 1.0])
    else:
        direction = np.asarray(direction, dtype=float)
        direction = direction / np.linalg.norm(direction)

    copies = []
    for i in range(n_copies):
        position = start + i * separation * direction
        copy = mesh.copy()
        copy.points += position - np.array(mesh.center)
        copies.append(copy)

    copies = merge_meshes(copies)  # Combine into a single mesh

    return copies
