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
)
from membrain_stats.utils.wrt_utils import get_wrt_inputs

# from membrain_stats.utils.geodesic_distance_utils import (
#     compute_geodesic_distance_matrix,
# )
from membrain_stats.geodesic_distances.geodesic_nearest_neighbors import (
    geodesic_nearest_neighbors,
)
from membrain_stats.membrane_edges.edge_from_curvature import exclude_edges_from_mesh


# def geodesic_nearest_neighbors(
#     verts: np.ndarray,
#     faces: np.ndarray,
#     point_coordinates: np.ndarray,
#     point_coordinates_target: np.ndarray,
#     method: str = "fast",
#     num_neighbors: int = 1,
# ):
#     """
#     Compute the geodesic nearest neighbors for a single mesh.

#     Parameters
#     ----------
#     verts : np.ndarray
#         The vertices of the mesh.
#     faces : np.ndarray
#         The faces of the mesh.
#     point_coordinates : np.ndarray
#         The coordinates of the start points.
#     point_coordinates_target : np.ndarray
#         The coordinates of the target points.
#     method : str
#         The method to use for computing geodesic distances. Can be either "exact" or "fast".
#     num_neighbors : int
#         The number of nearest neighbors to consider.
#     """
#     distance_matrix = compute_geodesic_distance_matrix(
#         verts=verts,
#         faces=faces,
#         point_coordinates=point_coordinates,
#         point_coordinates_target=point_coordinates_target,
#         method=method,
#     )
#     # replace -1 with inf
#     distance_matrix[distance_matrix == -1] = np.inf
#     nearest_neighbor_indices = np.argsort(distance_matrix, axis=1)[:, :num_neighbors]
#     nearest_neighbor_distances = np.sort(distance_matrix, axis=1)[:, :num_neighbors]

#     # pad with -1 if less than num_neighbors
#     nearest_neighbor_indices = np.pad(
#         nearest_neighbor_indices,
#         ((0, 0), (0, num_neighbors - nearest_neighbor_indices.shape[1])),
#         constant_values=-1,
#     )
#     nearest_neighbor_distances = np.pad(
#         nearest_neighbor_distances,
#         ((0, 0), (0, num_neighbors - nearest_neighbor_distances.shape[1])),
#         constant_values=-1,
#     )

#     return nearest_neighbor_indices, nearest_neighbor_distances


def geodesic_nearest_neighbors_wrt_folder(
    in_folder: str,
    out_folder: str,
    exclude_edges: bool = False,
    edge_exclusion_width: float = 50.0,
    pixel_size_multiplier: float = None,
    num_neighbors: int = 1,
    start_classes: List[int] = [0],
    target_classes: List[int] = [0],
    with_respect_to_class: int = 0,
    num_bins: int = 25,
    geod_distance_method: str = "fast",
    distance_matrix_method: str = "geodesic",
):
    """ """
    filenames = get_mesh_filenames(in_folder)

    mesh_dicts = [
        get_mesh_from_file(filename, pixel_size_multiplier=pixel_size_multiplier)
        for filename in filenames
    ]
    if exclude_edges:
        mesh_dicts = [
            exclude_edges_from_mesh(
                out_folder=out_folder,
                filename=filename,
                mesh_dict=mesh_dict,
                edge_exclusion_width=edge_exclusion_width,
                leave_classes=[with_respect_to_class],
            )
            for filename, mesh_dict in zip(filenames, mesh_dicts)
        ]
    mesh_dicts = [
        get_geodesic_distance_input(mesh_dict, start_classes, target_classes)
        for mesh_dict in mesh_dicts
    ]

    nn_data = [
        geodesic_nearest_neighbors(
            verts=mesh_dicts[i]["verts"],
            faces=mesh_dicts[i]["faces"],
            point_coordinates=mesh_dicts[i]["positions_start"],
            point_coordinates_target=mesh_dicts[i]["positions_target"],
            method=geod_distance_method,
            num_neighbors=num_neighbors,
        )
        for i in range(len(mesh_dicts))
    ]
    nearest_neighbor_distances = [data[1] for data in nn_data]

    # get WRT inputs
    meshes = [
        trimesh.Trimesh(
            vertices=mesh_dict["verts"],
            faces=mesh_dict["faces"],
        )
        for mesh_dict in mesh_dicts
    ]

    protein_nearest_wrt_distances, _, _ = get_wrt_inputs(
        mesh_dicts=mesh_dicts,
        meshes=meshes,
        consider_classes=start_classes,
        with_respect_to_class=with_respect_to_class,
        geod_distance_method=geod_distance_method,
        distance_matrix_method=distance_matrix_method,
    )

    bins = np.linspace(0, np.max(protein_nearest_wrt_distances), num_bins)
    all_nearest_neighbor_distances = np.concatenate(nearest_neighbor_distances)
    y_data = []
    for bin0, bin1 in zip(bins[:-1], bins[1:]):
        mask = (protein_nearest_wrt_distances >= bin0) & (
            protein_nearest_wrt_distances < bin1
        )
        y_data.append(np.mean(all_nearest_neighbor_distances[mask]))

    # store output
    out_data = {
        "bin_lower": bins[:-1],
        "bin_upper": bins[1:],
        "mean_nearest_neighbor_distance": y_data,
    }
    out_data = pd.DataFrame(out_data)
    out_file = os.path.join(out_folder, "nearest_neighbor_distances_wrt_bins.star")
    starfile.write(out_data, out_file)

    # store image
    from matplotlib import pyplot as plt

    plt.figure()
    plt.plot(bins[:-1], y_data)
    plt.xlabel("Distance to nearest protein")
    plt.ylabel("Mean distance to nearest neighbor")
    plt.savefig("nearest_neighbor_distances_wrt_bins.png")
    plt.close()
