import numpy as np
import math

def area_cylinder(radius, height):
    """
    Calculate the surface area and lateral area of a cylinder.
    
    Parameters:
    radius (float): Radius of the cylinder
    height (float): Height of the cylinder
    
    Returns:
    tuple: (lateral_area, total_surface_area)
    """
    lateral_area = 2 * np.pi * radius * height
    base_area = np.pi * radius**2
    total_surface_area = lateral_area + 2 * base_area
    return total_surface_area

def volume_cylinder(radius, height):
    """
    Calculate the volume of a cylinder.
    
    Parameters:
    radius (float): Radius of the cylinder
    height (float): Height of the cylinder
    
    Returns:
    float: Volume of the cylinder
    """
    return np.pi * radius**2 * height

def helix_pitch_from_tilt(radius, tilt_angle_deg):
    """
    Calculate the helix pitch (axial rise per full turn) 
    given its radius and tilt angle above the horizontal.

    Parameters:
        radius (float): Helix radius R.
        tilt_angle_deg (float): Tilt angle α above horizontal, in degrees.

    Returns:
        float: Pitch P (axial rise per full 360° turn).
    """
    alpha = math.radians(tilt_angle_deg)
    return 2 * math.pi * radius * math.tan(alpha)


def helix_radius_from_tilt(pitch, tilt_angle_deg):
    """
    Calculate the helix radius required to achieve a given tilt angle and pitch.

    Parameters:
        pitch (float): Helix pitch P (axial rise per full turn).
        tilt_angle_deg (float): Desired tilt angle α above the horizontal, in degrees.

    Returns:
        float: Required helix radius R.
    """
    # Convert degrees to radians
    alpha = math.radians(tilt_angle_deg)
    # R = P / (2π · tan(α))
    return pitch / (2 * math.pi * math.tan(alpha))

def radius_cone_from_tilt(pitch, tilt_angle_deg):
    """
    Calculate the radius of a cone required to achieve a given tilt angle and pitch.

    Parameters:
        pitch (float): Helix pitch P (axial rise per full turn).
        tilt_angle_deg (float): Desired tilt angle α above the horizontal, in degrees.

    Returns:
        float: Required cone radius R.
    """
    # Convert degrees to radians
    alpha = math.radians(tilt_angle_deg)
    # R = P / (2π · tan(α))
    return pitch / (2 * math.pi * math.tan(alpha))

def calculate_helix_length(radius=1.0, pitch=2.0, turns=3.0):
    """
    Calculate the exact arc length of a helix analytically.
    
    Parameters
    ----------
    radius : float
        Radius of the helix
    pitch : float
        Vertical distance per complete turn
    turns : float
        Number of complete turns
    
    Returns
    -------
    length : float
        Total arc length of the helix
    """
    # Arc length formula for helix: L = turns * sqrt((2*pi*r)^2 + pitch^2)
    circumference = 2 * np.pi * radius
    length_per_turn = np.sqrt(circumference**2 + pitch**2)
    total_length = turns * length_per_turn
    return total_length

def get_x_y_coordinantes_circle_angle(r, c1, angle):
    """
    Get the x and y coordinates of a point on a circle given its radius, center, and angle.
    
    Parameters:
    r (float): Radius of the circle.
    c1 (tuple): Center of the circle (x, y).
    angle (float): Angle in degrees.
    
    Returns:
    tuple: Coordinates of the point on the circle (x, y).
    """
    import math
    x = c1[0] + r * math.cos(math.radians(angle))
    y = c1[1] + r * math.sin(math.radians(angle))
    return (x, y)

def find_cicles_intersection(r1, r2, c1, c2):
    """
    Find the intersection points of two circles.
    
    Parameters:
    r1 (float): Radius of the first circle.
    r2 (float): Radius of the second circle.
    c1 (tuple): Center of the first circle (x1, y1).
    c2 (tuple): Center of the second circle (x2, y2).
    
    Returns:
    list: A list of intersection points as tuples (x, y).
    """
    import math
    
    x1, y1 = c1
    x2, y2 = c2
    
    d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    if d > r1 + r2 or d < abs(r1 - r2):
        return []  # No intersection
    
    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h = math.sqrt(r1**2 - a**2)
    
    x0 = x1 + a * (x2 - x1) / d
    y0 = y1 + a * (y2 - y1) / d
    
    intersection_points = []
    
    if h == 0:
        intersection_points.append((x0, y0))  # One point of intersection
    else:
        rx = -(y2 - y1) * (h / d)
        ry = -(x2 - x1) * (h / d)
        
        intersection_points.append((x0 + rx, y0 + ry))
        intersection_points.append((x0 - rx, y0 - ry))
    
    return intersection_points

