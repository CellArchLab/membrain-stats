from typing import List
import os
import numpy as np
import pandas as pd
import starfile
import trimesh

from membrain_stats.utils.io_utils import (
    get_mesh_filenames,
    get_mesh_from_file,
    get_tmp_edge_files,
)
from membrain_stats.membrane_edges.edge_from_curvature import (
    exclude_edges_from_mesh,
)
from membrain_stats.geodesic_distances.geodesic_distances import (
    compute_euclidean_distance_matrix,
    compute_geodesic_distance_matrix,
)
from membrain_stats.utils.mesh_utils import barycentric_area_per_vertex


def protein_concentration_wrt_folder(
    in_folder: str,
    out_folder: str,
    exclude_edges: bool = False,
    edge_exclusion_width: float = 50.0,
    pixel_size_multiplier: float = None,
    consider_classes: List[int] = "all",
    with_respect_to_class: int = 0,
    num_bins: int = 25,
    geod_distance_method: str = "exact",
    distance_matrix_method: str = "geodesic",
):

    filenames = get_mesh_filenames(in_folder)
    mesh_dicts = [
        get_mesh_from_file(filename, pixel_size_multiplier=pixel_size_multiplier)
        for filename in filenames
    ]
    meshes = [
        trimesh.Trimesh(
            vertices=mesh_dict["verts"],
            faces=mesh_dict["faces"],
        )
        for mesh_dict in mesh_dicts
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

    protein_classes = [mesh_dict["classes"] for mesh_dict in mesh_dicts]
    wrt_positions = [
        mesh_dict["positions"][mesh_dict["classes"] == with_respect_to_class]
        for mesh_dict in mesh_dicts
    ]
    if -1 in consider_classes:
        masks = [classes != with_respect_to_class for classes in protein_classes]
        edge_masks = [classes != -1 for classes in protein_classes]
        masks = [mask & edge_mask for mask, edge_mask in zip(masks, edge_masks)]
    else:
        masks = [np.isin(classes, consider_classes) for classes in protein_classes]
    consider_positions = [
        mesh_dict["positions"][mask] for mask, mesh_dict in zip(masks, mesh_dicts)
    ]

    if distance_matrix_method == "geodesic":
        distance_matrix_outputs = [
            compute_geodesic_distance_matrix(
                verts=mesh.vertices,
                faces=mesh.faces,
                point_coordinates=wrt_positions[i],
                point_coordinates_target=consider_positions[i],
                method=geod_distance_method,
                return_mesh_distances=True,
            )
            for i, mesh in enumerate(meshes)
        ]
    elif distance_matrix_method == "euclidean":
        distance_matrix_outputs = [
            compute_euclidean_distance_matrix(
                verts=mesh.vertices,
                point_coordinates=wrt_positions[i],
                point_coordinates_target=consider_positions[i],
                return_mesh_distances=True,
            )
            for i, mesh in enumerate(meshes)
        ]

    distance_matrices = [output[0] for output in distance_matrix_outputs]
    mesh_distances = [output[1] for output in distance_matrix_outputs]

    protein_nearest_wrt_distances = [
        np.min(distance_matrix, axis=1) for distance_matrix in distance_matrices
    ]

    mesh_barycentric_areas = [barycentric_area_per_vertex(mesh) for mesh in meshes]

    # sort protein distances
    protein_nearest_wrt_distances = np.concatenate(protein_nearest_wrt_distances)
    protein_nearest_wrt_distances = np.sort(protein_nearest_wrt_distances)

    mesh_barycentric_areas = [
        np.repeat(barycentric_area, len(mesh_dist))
        for barycentric_area, mesh_dist in zip(mesh_barycentric_areas, mesh_distances)
    ]
    # flatten and concatenate
    mesh_barycentric_areas = np.concatenate(mesh_barycentric_areas)
    mesh_distances = [np.ravel(mesh_dist) for mesh_dist in mesh_distances]
    mesh_distances = np.concatenate(mesh_distances, axis=0)

    bins = np.linspace(0, np.max(protein_nearest_wrt_distances), num_bins)
    x_data = np.histogram(protein_nearest_wrt_distances, bins=bins)[0]
    y_data = []
    for num_bin, protein_numbers in enumerate(x_data):
        bin_lower = bins[num_bin]
        bin_upper = bins[num_bin + 1]
        area_mask = (mesh_distances >= bin_lower) & (mesh_distances < bin_upper)
        y_data.append(protein_numbers / np.sum(mesh_barycentric_areas[area_mask]))

    from matplotlib import pyplot as plt

    plt.figure()
    plt.plot(bins[:-1], y_data)
    plt.xlabel("Distance to nearest protein")
    plt.ylabel("Area covered")
    plt.savefig("./protein_concentration.png")
