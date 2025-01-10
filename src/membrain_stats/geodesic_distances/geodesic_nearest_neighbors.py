import os
import starfile
import numpy as np
import pandas as pd
import trimesh
from typing import List

from membrain_stats.utils.io_utils import (
    get_mesh_filenames,
    get_mesh_from_file,
    get_geodesic_distance_input,
    get_tmp_edge_files,
)
from membrain_stats.utils.geodesic_distance_utils import (
    compute_geodesic_distance_matrix,
)
from membrain_stats.utils.pairwise_orientations_utils import angle_matrix
from membrain_stats.membrane_edges.exclude_edge_positions import (
    mask_geodesic_edge_positions,
)


def geodesic_nearest_neighbors(
    verts: np.ndarray,
    faces: np.ndarray,
    point_coordinates: np.ndarray,
    point_coordinates_target: np.ndarray,
    method: str = "fast",
    num_neighbors: int = 1,
    angles_start: np.ndarray = None,
    angles_target: np.ndarray = None,
    compute_angles: bool = False,
    c2_symmetry: bool = False,
    project_to_plane: bool = False,
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
    angles_start : np.ndarray
        The angles of the start points.
    angles_target : np.ndarray
        The angles of the target points.
    compute_angles : bool
        Whether to compute the angles between the start and target points.
    c2_symmetry : bool
        Whether the protein has C2 symmetry.
    project_to_plane : bool
        Whether to project the mesh to a plane.

    """
    distance_matrix = compute_geodesic_distance_matrix(
        verts=verts,
        faces=faces,
        point_coordinates=point_coordinates,
        point_coordinates_target=point_coordinates_target,
        method=method,
    )
    nearest_neighbor_indices = np.argsort(distance_matrix, axis=1)[
        :, 1 : num_neighbors + 1
    ]  # start from 1 to exclude the point itself
    nearest_neighbor_distances = np.sort(distance_matrix, axis=1)[
        :, 1 : num_neighbors + 1
    ]

    # pad with -1 if less than num_neighbors
    nearest_neighbor_indices = np.pad(
        nearest_neighbor_indices,
        ((0, 0), (0, num_neighbors - nearest_neighbor_indices.shape[1])),
        constant_values=-1,
    )
    nearest_neighbor_distances = np.pad(
        nearest_neighbor_distances,
        ((0, 0), (0, num_neighbors - nearest_neighbor_distances.shape[1])),
        constant_values=-1,
    )

    if compute_angles:
        angles = angle_matrix(
            angles1=angles_start,
            angles2=angles_target,
            c2_symmetry=c2_symmetry,
            pos1=point_coordinates,
            pos2=point_coordinates_target,
            mesh=trimesh.Trimesh(vertices=verts, faces=faces),
            project_to_plane=project_to_plane,
        )
        nearest_neighbor_angles = np.take_along_axis(
            angles, nearest_neighbor_indices, axis=1
        )
    else:
        nearest_neighbor_angles = None

    return nearest_neighbor_indices, nearest_neighbor_distances, nearest_neighbor_angles


def geodesic_nearest_neighbors_singlemb(
    filename: str,
    out_file_edge: str,
    pixel_size_multiplier: float = None,
    pixel_size_multiplier_positions: float = None,
    num_neighbors: int = 1,
    start_classes: List[int] = [0],
    target_classes: List[int] = [0],
    method: str = "fast",
    c2_symmetry: bool = False,
    project_to_plane: bool = False,
    exclude_edges: bool = False,
    edge_exclusion_width: float = 50.0,
    edge_percentile: float = 95,
    store_sanity_meshes: bool = False,
):
    """
    Compute the geodesic nearest neighbors for a single mesh.

    Parameters
    ----------
    filename : str
        The filename of the mesh.
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
    mesh_dict = get_mesh_from_file(
        filename,
        pixel_size_multiplier=pixel_size_multiplier,
        pixel_size_multiplier_positions=pixel_size_multiplier_positions,
    )
    edge_exclusion_params = {
        "exclude_edges": exclude_edges,
        "edge_exclusion_width": edge_exclusion_width,
        "edge_percentile": edge_percentile,
        "out_file": out_file_edge,
        "store_sanity_meshes": store_sanity_meshes,
    }

    mesh_dict = get_geodesic_distance_input(
        mesh_dict,
        start_classes,
        target_classes,
    )

    if edge_exclusion_params["exclude_edges"]:
        print("Excluding edges")
        edge_mask = mask_geodesic_edge_positions(
            positions=mesh_dict["positions"],
            mesh_dict=mesh_dict,
            edge_exclusion_params=edge_exclusion_params,
            return_pos_mask=True,
        )
        mesh_dict["positions_start"] = mesh_dict["positions_start"][edge_mask]
        mesh_dict["angles_start"] = mesh_dict["angles_start"][edge_mask]

    nn_data = geodesic_nearest_neighbors(
        verts=mesh_dict["verts"],
        faces=mesh_dict["faces"],
        point_coordinates=mesh_dict["positions_start"],
        point_coordinates_target=mesh_dict["positions_target"],
        method=method,
        num_neighbors=num_neighbors,
        angles_start=mesh_dict["angles_start"],
        angles_target=mesh_dict["angles_target"],
        compute_angles=mesh_dict["hasAngles"],
        c2_symmetry=c2_symmetry,
        project_to_plane=project_to_plane,
    )
    nearest_neighbor_indices = nn_data[0]
    nearest_neighbor_distances = nn_data[1]
    nearest_neighbor_angles = nn_data[2]

    return (
        mesh_dict,
        nearest_neighbor_indices,
        nearest_neighbor_distances,
        nearest_neighbor_angles,
    )


def geodesic_nearest_neighbors_folder(
    in_folder: str,
    out_folder: str,
    pixel_size_multiplier: float = None,
    pixel_size_multiplier_positions: float = None,
    num_neighbors: int = 1,
    start_classes: List[int] = [0],
    target_classes: List[int] = [0],
    method: str = "fast",
    c2_symmetry: bool = False,
    project_to_plane: bool = False,
    plot: bool = False,
    exclude_edges: bool = False,
    edge_exclusion_width: float = 50.0,
    edge_percentile: float = 95,
    store_sanity_meshes: bool = False,
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
    out_files_edges = get_tmp_edge_files(out_folder, filenames)
    nn_outputs = [
        geodesic_nearest_neighbors_singlemb(
            filename,
            out_file_edge,
            pixel_size_multiplier=pixel_size_multiplier,
            pixel_size_multiplier_positions=pixel_size_multiplier_positions,
            num_neighbors=num_neighbors,
            start_classes=start_classes,
            target_classes=target_classes,
            method=method,
            c2_symmetry=c2_symmetry,
            project_to_plane=project_to_plane,
            exclude_edges=exclude_edges,
            edge_exclusion_width=edge_exclusion_width,
            edge_percentile=edge_percentile,
            store_sanity_meshes=store_sanity_meshes,
        )
        for filename, out_file_edge in zip(filenames, out_files_edges)
    ]

    mesh_dicts = [data[0] for data in nn_outputs]
    nearest_neighbor_indices = [data[1] for data in nn_outputs]
    nearest_neighbor_distances = [data[2] for data in nn_outputs]
    nearest_neighbor_angles = [data[3] for data in nn_outputs]

    # create a separate star file for each mesh
    for i in range(len(nearest_neighbor_indices)):
        out_data = {
            "filename": filenames[i],
            "start_positionX": np.array(mesh_dicts[i]["positions_start"][:, 0]),
            "start_positionY": np.array(mesh_dicts[i]["positions_start"][:, 1]),
            "start_positionZ": np.array(mesh_dicts[i]["positions_start"][:, 2]),
        }
        if mesh_dicts[i]["hasAngles"]:
            out_data["start_angleRot"] = np.array(mesh_dicts[i]["angles_start"][:, 0])
            out_data["start_angleTilt"] = np.array(mesh_dicts[i]["angles_start"][:, 1])
            out_data["start_anglePsi"] = np.array(mesh_dicts[i]["angles_start"][:, 2])
        for j in range(num_neighbors):
            out_data[f"nn{j}_positionX"] = np.array(
                mesh_dicts[i]["positions_target"][nearest_neighbor_indices[i][:, j], 0]
            )
            out_data[f"nn{j}_positionY"] = np.array(
                mesh_dicts[i]["positions_target"][nearest_neighbor_indices[i][:, j], 1]
            )
            out_data[f"nn{j}_positionZ"] = np.array(
                mesh_dicts[i]["positions_target"][nearest_neighbor_indices[i][:, j], 2]
            )
            if mesh_dicts[i]["hasAngles"]:
                out_data[f"nn{j}_angleRot"] = np.array(
                    mesh_dicts[i]["angles"][nearest_neighbor_indices[i][:, j], 0]
                )
                out_data[f"nn{j}_angleTilt"] = np.array(
                    mesh_dicts[i]["angles"][nearest_neighbor_indices[i][:, j], 1]
                )
                out_data[f"nn{j}_anglePsi"] = np.array(
                    mesh_dicts[i]["angles"][nearest_neighbor_indices[i][:, j], 2]
                )
            if nearest_neighbor_angles[i] is not None:
                out_data[f"nn{j}_angle"] = np.array(nearest_neighbor_angles[i][:, j])

            out_data[f"nn{j}_distance"] = np.array(nearest_neighbor_distances[i][:, j])
        out_data = pd.DataFrame(out_data)
        out_token = os.path.basename(filenames[i]).split(".")[0]
        out_file = os.path.join(out_folder, f"{out_token}_nearest_neighbors.star")
        os.makedirs(out_folder, exist_ok=True)
        starfile.write(out_data, out_file)

    if plot:
        # plot the nearest neighbors as histograms
        # concatenate all distances
        all_nearest_neighbor_distances = np.concatenate(nearest_neighbor_distances)
        from matplotlib import pyplot as plt

        plt.figure()
        plt.hist(all_nearest_neighbor_distances, bins=30)
        plt.xlabel("Nearest neighbor distance")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(out_folder, "nearest_neighbor_distances.png"))
