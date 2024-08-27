import os 
import starfile
import numpy as np
import pandas as pd
from typing import List

from membrain_stats.utils.io_utils import get_mesh_filenames, get_mesh_from_file
from membrain_stats.geodesic_distances import compute_geodesic_distance_matrix

def geodesic_nearest_neighbors(
    verts: np.ndarray,
    faces: np.ndarray,
    point_coordinates: np.ndarray,
    point_coordinates_target: np.ndarray,
    method: str = "fast",
    num_neighbors: int = 1,
):
    """
    Compute the geodesic nearest neighbors for a single mesh.

    Parameters
    ----------
    verts : np.ndarray
        The vertices of the mesh.
    faces : np.ndarray
        The faces of the mesh.
    point_coordinates : np.ndarray
        The coordinates of the start points.
    point_coordinates_target : np.ndarray
        The coordinates of the target points.
    method : str
        The method to use for computing geodesic distances. Can be either "exact" or "fast".
    num_neighbors : int
        The number of nearest neighbors to consider.
    """
    distance_matrix = compute_geodesic_distance_matrix(
        verts=verts,
        faces=faces,
        point_coordinates=point_coordinates,
        point_coordinates_target=point_coordinates_target,
        method=method,
    )
    nearest_neighbor_indices = np.argsort(distance_matrix, axis=1)[:, :num_neighbors]
    nearest_neighbor_distances = np.sort(distance_matrix, axis=1)[:, :num_neighbors]

    # pad with -1 if less than num_neighbors
    nearest_neighbor_indices = np.pad(nearest_neighbor_indices, ((0, 0), (0, num_neighbors - nearest_neighbor_indices.shape[1])), constant_values=-1)
    nearest_neighbor_distances = np.pad(nearest_neighbor_distances, ((0, 0), (0, num_neighbors - nearest_neighbor_distances.shape[1])), constant_values=-1)

    return nearest_neighbor_indices, nearest_neighbor_distances


def geodesic_nearest_neighbors_folder(
        in_folder: str,
        out_folder: str,
        pixel_size_multiplier: float = None,
        num_neighbors: int = 1,
        start_classes: List[int] = [0],
        target_classes: List[int] = [0],
        method: str = "fast",
):
    """
    Compute the geodesic nearest neighbors for all meshes in a folder.

    Parameters
    ----------
    in_folder : str
        The folder containing the meshes.
    out_folder : str
        The folder where the output star files should be stored.
    pixel_size_multiplier : float
        The pixel size multiplier if the mesh is not scaled in unit Angstrom. If provided, mesh vertices are multiplied by this value.
    num_neighbors : int
        The number of nearest neighbors to consider.
    start_classes : List[int]
        The list of classes to consider for start points.
    target_classes : List[int]
        The list of classes to consider for target points.
    method : str
        The method to use for computing geodesic distances. Can be either "exact" or "fast".
    """
    filenames = get_mesh_filenames(in_folder)
    mesh_dicts = [get_mesh_from_file(filename) for filename in filenames]
    vertices = [mesh_dict["verts"] for mesh_dict in mesh_dicts]
    vertices = [vertices[i] * (1.0 if pixel_size_multiplier is None else pixel_size_multiplier) for i in range(len(vertices))]
    faces = [mesh_dict["faces"] for mesh_dict in mesh_dicts]
    positions = [mesh_dict["positions"] for mesh_dict in mesh_dicts]
    classes = [mesh_dict["classes"] for mesh_dict in mesh_dicts]
    class_start_masks = [np.isin(classes[i], start_classes) for i in range(len(classes))]
    class_target_masks = [np.isin(classes[i], target_classes) for i in range(len(classes))]
    positions_start = [positions[i][class_start_masks[i]] for i in range(len(positions))]
    positions_target = [positions[i][class_target_masks[i]] for i in range(len(positions))]

    nn_data = [
        geodesic_nearest_neighbors(
            verts=vertices[i],
            faces=faces[i],
            point_coordinates=positions_start[i],
            point_coordinates_target=positions_target[i],
            method=method,
            num_neighbors=num_neighbors,
        ) for i in range(len(vertices))
    ]
    nearest_neighbor_indices = [data[0] for data in nn_data]
    nearest_neighbor_distances = [data[1] for data in nn_data]

    # create a separate star file for each mesh
    for i in range(len(nearest_neighbor_indices)):
        out_data = {
            "filename": filenames[i],
            "start_positionX": np.array(positions_start[i][:, 0]),
            "start_positionY": np.array(positions_start[i][:, 1]),
            "start_positionZ": np.array(positions_start[i][:, 2]),
        }
        for j in range(num_neighbors):
            out_data[f"nn{j}_positionX"] = np.array(positions_target[i][nearest_neighbor_indices[i][:, j], 0])
            out_data[f"nn{j}_positionY"] = np.array(positions_target[i][nearest_neighbor_indices[i][:, j], 1])
            out_data[f"nn{j}_positionZ"] = np.array(positions_target[i][nearest_neighbor_indices[i][:, j], 2])
            out_data[f"nn{j}_distance"] =  np.array(nearest_neighbor_distances[i][:, j])
        out_data = pd.DataFrame(out_data)
        out_token = os.path.basename(filenames[i]).split(".")[0]
        out_file = os.path.join(out_folder, f"{out_token}_nearest_neighbors.star")
        os.makedirs(out_folder, exist_ok=True)
        starfile.write(out_data, out_file)



