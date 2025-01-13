import numpy as np

import numpy as np
from scipy.spatial.transform import Rotation

def calculate_angle(p1, p2, p3):
    # Create vectors
    vector1 = np.array(p1) - np.array(p2)  
    vector2 = np.array(p3) - np.array(p2)  
    
    # Calculate the dot product and magnitudes
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    # Calculate the angle in radians and convert to degrees
    angle_radians = np.arccos(dot_product / (magnitude1 * magnitude2))
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees

def vector_side_angle(a,b,c,d):
    v1 = calculate_vector(a,b)
    v2 = calculate_vector(c,d)
    angle = vector_angle(v1,v2)
    return angle

def trunk_axial_rotation(Rs, Ls, Rh, Lh):
    # Project points onto the XZ plane (ignore Y-axis)
    Rs_xz = np.array([Rs[0], Rs[2]])  # Right shoulder
    Ls_xz = np.array([Ls[0], Ls[2]])  # Left shoulder
    Rh_xz = np.array([Rh[0], Rh[2]])  # Right hip
    Lh_xz = np.array([Lh[0], Lh[2]])  # Left hip

    # Calculate vectors in the XZ plane
    vector1 = Rs_xz - Ls_xz  # Shoulder vector in XZ
    vector2 = Rh_xz - Lh_xz  # Hip vector in XZ

    # Calculate dot product and norms
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    # Avoid division by zero
    if norm1 == 0 or norm2 == 0:
        raise ValueError("One of the vectors has zero magnitude, unable to compute angle.")

    # Calculate cosine of the angle
    cos_theta = dot_product / (norm1 * norm2)

    # Clamp to avoid numerical errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # Calculate the angle in radians and convert to degrees
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    # print("angle_deg: ", angle_deg)

    return angle_deg

######################################################################################################
def calculate_vector(start, end):
    """Calculate a vector from two 3D points."""
    return np.array(end) - np.array(start)

def vector_angle(v1, v2):
    """Calculate the angle (in degrees) between two vectors."""
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

def trunk_flexion(pelvis, mid_spine):
    trunk_vector = calculate_vector(pelvis, mid_spine)
    vertical_vector = np.array([0, 1, 0])  # Y-axis as vertical
    # sagittal_projection = np.array([trunk_vector[0], trunk_vector[1], trunk_vector[2]])  # Y-Z plane
    return vector_angle(trunk_vector, vertical_vector)
