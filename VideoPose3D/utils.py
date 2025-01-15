import numpy as np
import requests
import json
import os
from dotenv import load_dotenv
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()
BASE_URL = os.environ['BASE_URL']

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

def points_to_angles(points_list):
    """
    Compute the angle between vectors formed by points.
    """
    # Ensure all points are numpy arrays of shape (2,)
    try:
        points_array = np.array([np.array(pt[:2], dtype=float) for pt in points_list])
    except ValueError as e:
        print(f"Error converting points: {e}")
        return np.nan  # Return NaN on conversion error
    
    if len(points_list) == 3:  # For angle between vectors ba and bc
        vector_u = points_array[0] - points_array[1]  # Vector ba
        vector_v = points_array[2] - points_array[1]  # Vector bc
        ang = np.arctan2(vector_u[1], vector_u[0]) - np.arctan2(vector_v[1], vector_v[0])
    else:
        return np.nan  # Invalid number of points
    
    ang_deg = np.degrees(ang)
    return ang_deg if ang_deg >= 0 else ang_deg + 360  # Ensure positive angle


def create_reba_analysis(video_uuid, json_data, file_path):

    url = f"{BASE_URL}operation/reba/analysis/create/" 
    headers = {
        "Content-Type": "application/json",
    }
    data = {
        "video_uuid": str(video_uuid),  # video_uuid is a string
        "result": json_data,
        "processed_file_path": str(file_path)
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        logger.error(response.text)
        response.raise_for_status() 
        
        return response.json()  

    except requests.exceptions.RequestException as e:
        print(f"Error calling REBA analysis API: {e}")
        return None 

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
        return None
