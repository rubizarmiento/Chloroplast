"""
intersections.py — Connector geometry between right- and left-handed helices.

Step 1: outer splines, circle-circle intersections, 2D plot.
Step 2: connector loop points.
Step 3: filled + extruded connector mesh.
"""

import io
import contextlib
import math
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

from .pyvista_geometries import generate_helix_spline, create_elliptical_helix, line_between_points, fill_nonplanar_loop_with_grid
from .simple_math_geometry import find_cicles_intersection, get_x_y_coordinantes_circle_angle
from .mesh_utils import mesh_resolution
from .assembly import build_rotational_array
from .builders import get_right_helix_parameters, get_left_helix_parameters


# ---------------------------------------------------------------------------
# Step 1 — outer splines, intersections, plot
# ---------------------------------------------------------------------------

def get_outer_right_spline(right_params, n_points=1001):
    """
    Generate the outer centerline spline for the right-handed helix (single copy, at origin).

    Parameters
    ----------
    right_params : dict
        Output of get_right_helix_parameters(). Uses radius_spline, pitch, n_turns.
    n_points : int
        Number of points along the spline.

    Returns
    -------
    pyvista.PolyData
        Spline points centered in z (z-shifted by height_crosssection / 2).
    """
    points = generate_helix_spline(
        x0=0, y0=0, z0=0,
        radius=right_params['radius_spline'],
        pitch=right_params['pitch'],
        turns=right_params['n_turns'],
        n_points=n_points,
        chirality='right',
    )
    spline = pv.PolyData(points)
    # Shift up by half the cross-section height so the spline sits on the
    # top surface of the tube (tube extends ±height_crosssection/2 around its centerline).
    spline.points[:, 2] -= right_params['height_crosssection'] / 2
    return spline


def get_outer_left_spline(left_params, n_points=1001):
    """
    Generate the outer centerline spline for one left-handed helix (at its nominal position).

    Parameters
    ----------
    left_params : dict
        Output of get_left_helix_parameters(). Uses radius_spline, pitch, n_turns, d_granum_helix.
    n_points : int
        Number of points along the spline.

    Returns
    -------
    pyvista.PolyData
        Spline points placed at [0, -d_granum_helix, 0].
    """
    points = create_elliptical_helix(
        x0=0, y0=-left_params['d_granum_helix'], z0=0,
        radius_x=left_params['radius_spline'],
        radius_y=left_params['radius_spline'],
        pitch=left_params['pitch'],
        turns=left_params['n_turns'],
        n_points=n_points,
        chirality='left',
    )
    spline = pv.PolyData(points)
    # Shift up by half the cross-section height so the spline sits on the
    # top surface of the tube (tube extends ±height_crosssection/2 around its centerline).
    spline.points[:, 2] += left_params['height_crosssection'] / 2
    return spline


def get_helix_intersections(right_params, left_params):
    """
    Compute the circle-circle intersection between the right and left helix outer radii,
    and return the key geometric points used to define the connector loop.

    Parameters
    ----------
    right_params : dict
        Output of get_right_helix_parameters(). Uses total_radius_helix, radius_spline.
    left_params : dict
        Output of get_left_helix_parameters(). Uses total_radius_helix, d_granum_helix, radius_spline.

    Returns
    -------
    intersections : list of tuple
        (x, y) circle-circle intersection points.
    key_points : dict
        x1/y1  — point on left helix spline at angle 0°
        x4/y4  — point on right helix spline at angle 45°
        x4_/y4_ — x4/y4 offset inward by height_crosssection / 2
    """
    d = left_params['d_granum_helix']

    intersections = find_cicles_intersection(
        r1=left_params['total_radius_helix'],
        r2=right_params['total_radius_helix'],
        c1=(d, 0),
        c2=(0, 0),
    )

    x1, y1 = get_x_y_coordinantes_circle_angle(
        left_params['radius_spline'], c1=(d, 0), angle=0
    )
    x4, y4 = get_x_y_coordinantes_circle_angle(
        right_params['radius_spline'], c1=(0, 0), angle=45
    )
    hc = right_params['height_crosssection']
    x4_ = x4 - hc / 2
    y4_ = y4 + hc / 2

    # x2: bridge point on the far side of the left helix from x1
    x2 = x1 + 2 * left_params['total_radius_helix']
    y2 = y1

    # x3: x2 projected onto the 45° direction toward the right helix
    x3 = (x2 / math.sqrt(2)) * math.cos(math.radians(45))
    y3 = (x2 / math.sqrt(2)) * math.sin(math.radians(45))
    x3_ = x3 - hc / 2
    y3_ = y3 + hc / 2

    # x5: intersection point shifted inward by half the right helix tube width
    x5 = intersections[0][0] - right_params['width_helix'] / 2
    y5 = intersections[0][1]

    key_points = dict(
        x1=x1, y1=y1,
        x2=x2, y2=y2,
        x3=x3, y3=y3, x3_=x3_, y3_=y3_,
        x4=x4, y4=y4, x4_=x4_, y4_=y4_,
        x5=x5, y5=y5,
    )
    return intersections, key_points


def plot_helix_intersections(outer_right, outer_left_copies, intersections, key_points):
    """
    2D projection (XY plane) of the outer splines, intersection points, and key geometry.

    Parameters
    ----------
    outer_right : pyvista.PolyData
        Single right outer spline (un-rotated).
    outer_left_copies : list of pyvista.PolyData
        Rotational copies of the left outer spline.
    intersections : list of tuple
        Output of get_helix_intersections().
    key_points : dict
        Output of get_helix_intersections().

    Returns
    -------
    fig, ax : matplotlib Figure and Axes.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # outer_left_copies may be a merged PolyData (from build_rotational_array)
    # or a list — handle both
    if isinstance(outer_left_copies, list):
        for i, spline in enumerate(outer_left_copies):
            ax.plot(spline.points[:, 0], spline.points[:, 1], color='purple',
                    linewidth=2, label='Left helix' if i == 0 else None)
    else:
        ax.plot(outer_left_copies.points[:, 0], outer_left_copies.points[:, 1],
                color='purple', linewidth=2, label='Left helix')

    ax.plot(outer_right.points[:, 0], outer_right.points[:, 1],
            color='blue', linewidth=2, label='Right helix')

    if intersections:
        ax.plot(intersections[0][0], intersections[0][1],
                'ro', markersize=8, label='Intersection')

    kp = key_points
    ax.plot(kp['x1'],  kp['y1'],  'go', markersize=8, label='x1 (left @ 0°)')
    ax.plot(kp['x2'],  kp['y2'],  'ro', markersize=8, label='x2 (far side of left)')
    ax.plot(kp['x3'],  kp['y3'],  'bo', markersize=8, label='x3 (right @ 45° proj)')
    ax.plot(kp['x3_'], kp['y3_'], 'bx', markersize=8, label='x3 offset')
    ax.plot(kp['x4'],  kp['y4'],  'yo', markersize=8, label='x4 (right @ 45°)')
    ax.plot(kp['x4_'], kp['y4_'], 'yx', markersize=8, label='x4 offset')
    ax.plot(kp['x5'],  kp['y5'],  'co', markersize=8, label='x5 (intersection shifted)')

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X (nm)')
    ax.set_ylabel('Y (nm)')
    ax.set_title('Helix outer splines and intersection points')
    ax.grid(True)
    ax.legend()
    return fig, ax


# ---------------------------------------------------------------------------
# Step 2 — connector loop points
# ---------------------------------------------------------------------------

def filter_spline_region(spline, x_limits, y_limits, z_limits=None):
    """
    Filter points from a spline that fall within a bounding box.

    Parameters
    ----------
    spline : pyvista.PolyData or np.ndarray
        Spline to filter.
    x_limits, y_limits : list of [min, max]
        Bounding limits along x and y.
    z_limits : list of [min, max] or None
        Bounding limits along z. If None, no z filtering is applied.

    Returns
    -------
    np.ndarray of shape (N, 3)
        Filtered points.
    """
    pts = spline.points if hasattr(spline, 'points') else np.asarray(spline)
    mask = (
        (pts[:, 0] >= x_limits[0]) & (pts[:, 0] <= x_limits[1]) &
        (pts[:, 1] >= y_limits[0]) & (pts[:, 1] <= y_limits[1])
    )
    if z_limits is not None:
        mask &= (pts[:, 2] >= z_limits[0]) & (pts[:, 2] <= z_limits[1])
    return pts[mask]


def _make_line(p1, p2, target_resolution):
    """Line segment from p1 to p2 with spacing ≈ target_resolution."""
    d = np.linalg.norm(np.array(p2) - np.array(p1))
    n = int(round(d / target_resolution)) + 1
    return line_between_points(p1, p2, n_segments=n)


def build_connector_loop(outer_right, outer_left, key_points, intersections,
                         z_limits, target_resolution=2):
    """
    Build the closed loop of points that defines the connector patch between
    the right- and left-handed helices.

    Parameters
    ----------
    outer_right : pyvista.PolyData
        Merged 4-copy rotational array of the right outer spline
        (use build_rotational_array on get_outer_right_spline output).
    outer_left : pyvista.PolyData
        Merged rotational copies of the left outer spline.
    key_points : dict
        Output of get_helix_intersections().
    intersections : list of tuple
        Output of get_helix_intersections().
    z_limits : list of [min, max]
        Z range used to filter both splines. Tweak this to place the
        connector at the correct height in the model.
    target_resolution : float
        Spacing between points on the connecting line segments (nm).

    Returns
    -------
    loop_polydata : pyvista.PolyData
        Closed loop with line connectivity, ready for fill_nonplanar_loop_with_grid.
    right_points : np.ndarray
        Filtered right spline points used in the loop.
    left_points : np.ndarray
        Filtered left spline points used in the loop.
    """
    kp    = key_points
    ix, iy = intersections[0]

    # --- Filter right spline: x/y + z (same z window as left)
    # Pass 4 rotational copies so all angular positions are represented at each z.
    right_points = filter_spline_region(
        outer_right,
        x_limits=[kp['x4'] - target_resolution, ix],
        y_limits=[iy, kp['y4'] + target_resolution],
        z_limits=z_limits,
    )

    # --- Filter left spline: x/y + z
    left_points = filter_spline_region(
        outer_left,
        x_limits=[ix + target_resolution, kp['x1']],
        y_limits=[kp['y1'] - target_resolution, iy],
        z_limits=z_limits,
    )

    # --- 3D bridge points (inherit z from right spline endpoints)
    point2 = [kp['x2'], kp['y2'], right_points[-1][2]]
    point3 = [kp['x3'], kp['y3'], right_points[-1][2]]

    # --- Connecting line segments
    left_to_point2   = _make_line(left_points[-1],  point2, target_resolution)
    point2_to_point3 = _make_line(point2,            point3, target_resolution)
    point3_to_right  = _make_line(point3,  right_points[-1], target_resolution)

    # --- Assemble loop: reversed right + left + three bridge segments
    reverse_right = np.vstack((right_points[::-1], left_points[0]))
    loop_segments = [reverse_right, left_points,
                     left_to_point2, point2_to_point3, point3_to_right]

    # --- Continuity check
    for i, seg in enumerate(loop_segments):
        next_seg = loop_segments[(i + 1) % len(loop_segments)]
        gap = np.linalg.norm(np.array(seg[-1]) - np.array(next_seg[0]))
        if gap > 1e-6:
            print(f"WARNING: gap between segment {i+1} and {(i+1)%len(loop_segments)+1}: "
                  f"{gap:.3f} nm")

    loop_pts = np.vstack(loop_segments)

    # --- Build pyvista PolyData with explicit line edges
    edges = [(i, i + 1) for i in range(len(loop_pts) - 1)]
    lines = np.hstack([[2, i, j] for i, j in edges]).astype(np.int64)
    loop_polydata = pv.PolyData(loop_pts, lines=lines)

    print(f"Loop: {len(right_points)} right pts, {len(left_points)} left pts, "
          f"{len(loop_pts)} total")

    return loop_polydata, right_points, left_points


# ---------------------------------------------------------------------------
# Step 3 — filled + extruded connector mesh
# ---------------------------------------------------------------------------

def build_connector_from_loop(dict_params, loop_polydata, target_resolution=0.5, verbose=True):
    """
    Build the connector mesh from a pre-computed loop.

    Wraps build_connector_mesh using dict_params for height_crosssection,
    so the loop-building and mesh-building steps can be run separately.

    Parameters
    ----------
    dict_params : dict
        Output of read_model_parameters().
    loop_polydata : pyvista.PolyData
        Output of build_connector_loop().
    target_resolution : float
        Target triangle area in nm² for the filled surface.
    verbose : bool
        Print mesh resolution stats when True.

    Returns
    -------
    connector_mesh : pyvista.PolyData
        Extruded connector patch.
    filled : pyvista.PolyData
        Flat filled surface before extrusion (useful for inspection).
    """
    return build_connector_mesh(
        loop_polydata,
        height_crosssection=dict_params['height_crosssection'],
        target_resolution=target_resolution,
        verbose=verbose,
    )


def build_connector_mesh(loop_polydata, height_crosssection, target_resolution=2, verbose=True):
    """
    Fill the connector loop with a triangular surface and extrude it to give
    it membrane thickness.

    Parameters
    ----------
    loop_polydata : pyvista.PolyData
        Output of build_connector_loop().
    height_crosssection : float
        Extrusion distance in nm along z — equals height_granum from model parameters.
    target_resolution : float
        Target triangle area in nm² for the filled surface.
    verbose : bool
        Print mesh resolution stats when True.

    Returns
    -------
    connector_mesh : pyvista.PolyData
        Extruded connector patch ready for export or boolean operations.
    filled : pyvista.PolyData
        The flat filled surface before extrusion (useful for inspection).
    """
    filled = fill_nonplanar_loop_with_grid(loop_polydata.points, target_area=target_resolution)

    if verbose:
        from .mesh_utils import mesh_resolution
        mesh_resolution(filled)

    connector_mesh = filled.extrude([0, 0, height_crosssection], capping=True)

    return connector_mesh, filled


def build_connector_thylakoid_mesh(dict_params, target_resolution=2, z_limits=None, verbose=True):
    """
    Build the single connector mesh that bridges right- and left-handed helices.

    Parameters
    ----------
    dict_params : dict
        Output of read_model_parameters().
    target_resolution : float
        Target triangle area in nm² for the filled connector surface.
    z_limits : list of [min, max], optional
        Z window used to filter spline points. Defaults to [-39.5, 39.5] nm.
    verbose : bool
        Print mesh statistics when True.

    Returns
    -------
    connector_mesh : pyvista.PolyData
        Extruded connector patch.
    filled : pyvista.PolyData
        Flat filled surface before extrusion (useful for inspection).
    """
    if z_limits is None:
        z_limits = [-39.5, 39.5] #Magic number, should be replaced by left-helix height (pitch) in the future.

    with contextlib.redirect_stdout(io.StringIO()):
        right_params = get_right_helix_parameters(dict_params)
        left_params  = get_left_helix_parameters(dict_params)

        outer_right       = get_outer_right_spline(right_params)
        outer_left        = get_outer_left_spline(left_params)
        outer_left_copies = build_rotational_array(outer_left, n_copies=4)
        outer_right_4     = build_rotational_array(outer_right, n_copies=4)

        intersections, key_points = get_helix_intersections(right_params, left_params)

        loop_polydata, _, _ = build_connector_loop(
            outer_right_4, outer_left_copies,
            key_points, intersections,
            z_limits=z_limits,
            target_resolution=2,
        )

        connector_mesh, filled = build_connector_mesh(
            loop_polydata,
            height_crosssection=dict_params['height_crosssection'],
            target_resolution=target_resolution,
            verbose=False,
        )

    if verbose:
        mesh_resolution(connector_mesh)


    return connector_mesh
