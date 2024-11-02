from pytorch3d.io import load_objs_as_meshes
import torch
import numpy as np


def parse_pose_data(pose_data, device="cuda"):
    """
    Extracts rotation (R) and translation (T) matrices from pose data.
    Args:
        pose_data (dict): Dictionary containing pose information.

    Returns:
        R (torch.Tensor): 3x3 rotation matrix.
        T (torch.Tensor): 1x3 translation vector.
    """
    # Extract rotation (R) and translation (T) from the pose data
    camera_rt_matrix = pose_data["camera_rt_matrix"]
    R = torch.tensor([row[:3] for row in camera_rt_matrix], dtype=torch.float32, device=device)
    T = torch.tensor([row[3] for row in camera_rt_matrix], dtype=torch.float32, device=device)

    K = torch.eye(4, device=device)
    K[:3,:3] = torch.tensor(pose_data["camera_k_matrix"], dtype=torch.float32, device=device)


    return R, T, K

def load_mesh(mesh_path, device="cuda"):
    """
    Loads a mesh from the given .obj file path and moves it to the specified device.

    Args:
        mesh_path (str): Path to the .obj mesh file.
        device (torch.device): Device to load the mesh on (CPU/GPU).

    Returns:
        Meshes: A PyTorch3D mesh object.
    """
    try:
        mesh = load_objs_as_meshes([mesh_path], device=device)
        print(f"Successfully loaded mesh: {mesh_path}")
        return mesh
    except Exception as e:
        print(f"Failed to load mesh from {mesh_path}: {e}")
        raise


