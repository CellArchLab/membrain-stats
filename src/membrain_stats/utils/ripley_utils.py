from typing import List
import numpy as np
import trimesh
from membrain_stats.utils.mesh_utils import barycentric_area_per_vertex
from membrain_stats.utils.geodesic_distance_utils import (
    compute_geodesic_distance_matrix,
)


def compute_ripleys_stats(
    mesh_dict: List[dict],
    method: str = "fast",
):
    distance_matrix, mesh_distances = compute_geodesic_distance_matrix(
        verts=mesh_dict["verts"],
        faces=mesh_dict["faces"],
        point_coordinates=mesh_dict["positions_start"],
        point_coordinates_target=mesh_dict["positions_target"],
        method=method,
        return_mesh_distances=True,
    )
    mesh = trimesh.Trimesh(vertices=mesh_dict["verts"], faces=mesh_dict["faces"])
    barycentric_areas = barycentric_area_per_vertex(mesh)
    return distance_matrix, mesh_distances, barycentric_areas


def get_ripleys_inputs(ripley_stats: List[dict]):
    distance_matrices = [ripley_stat[0] for ripley_stat in ripley_stats]
    mesh_distances = [ripley_stat[1] for ripley_stat in ripley_stats]
    barycentric_areas = [ripley_stat[2] for ripley_stat in ripley_stats]

    return distance_matrices, mesh_distances, barycentric_areas


def get_xaxis_distances(distance_matrices: List[np.array], num_bins: int):
    # flatten protein-protein distances
    all_distances = np.concatenate(
        [np.ravel(distance_matrix) for distance_matrix in distance_matrices]
    )
    all_distances = all_distances[all_distances != -1]  # exlude distance to self

    # sort protein-protein distances
    sort_indices = np.argsort(all_distances)
    all_distances = all_distances[sort_indices]

    # split distances into bins
    distance_histogram, bin_edges = np.histogram(
        all_distances[all_distances < np.inf], bins=num_bins
    )
    all_distances = bin_edges[:-1]

    return all_distances, distance_histogram


def get_number_of_points(distance_matrices: List[np.array]):
    # compute number of starting and reachable points
    num_starting_points = sum(
        [len(distance_matrix) for distance_matrix in distance_matrices]
    )
    avg_starting_points = num_starting_points / len(distance_matrices)

    num_reachable_points = [
        [np.sum(distance_matrix[:, i] < np.inf) for i in range(len(distance_matrix))]
        for distance_matrix in distance_matrices
    ]
    avg_reachable_points = [np.mean(entry) for entry in num_reachable_points]

    return avg_starting_points, avg_reachable_points


def get_barycentric_areas(
    barycentric_areas: List[np.array], mesh_distances: List[np.array]
):
    # compute barycentric areas for each vertex and shape corresponding to all_mesh_distances
    repeated_barycentric_areas = [
        np.repeat(barycentric_area, len(mesh_distance))
        for barycentric_area, mesh_distance in zip(barycentric_areas, mesh_distances)
    ]
    all_barycentric_areas = np.concatenate(
        [
            np.ravel(repeated_barycentric_area)
            for repeated_barycentric_area in repeated_barycentric_areas
        ]
    )
    return all_barycentric_areas


def sort_barycentric_areas_and_mesh_distances(
    all_mesh_distances: List[np.array],
    all_barycentric_areas: List[np.array],
):
    # sort barycentric areas and mesh distances by mesh distance
    sort_indices = np.argsort(all_mesh_distances)
    all_mesh_distances = all_mesh_distances[sort_indices]
    all_barycentric_areas = all_barycentric_areas[sort_indices]
    return all_mesh_distances, all_barycentric_areas


def accumulate_barycentric_areas(
    all_barycentric_areas: List[np.array],
    all_mesh_distances: List[np.array],
    all_distances: List[np.array],
    protein_per_distance: np.array,
    ripley_type: str,
):
    # compute barycentric areas for each x-axis split (i.e. all_distances)
    x_barycentric_areas = np.split(
        all_barycentric_areas, np.searchsorted(all_mesh_distances, all_distances)
    )[:-1]
    x_barycentric_areas = np.array(
        [np.sum(x_barycentric_area) for x_barycentric_area in x_barycentric_areas]
    )

    total_conc = np.sum(protein_per_distance) / np.sum(x_barycentric_areas)

    # accumulate if not computing O statistic
    if ripley_type != "O":
        protein_per_distance = np.cumsum(protein_per_distance)
        x_barycentric_areas = np.cumsum(x_barycentric_areas)

    return protein_per_distance, x_barycentric_areas, total_conc


def define_xy_values(
    all_distances: np.array,
    protein_per_distance: np.array,
    x_barycentric_areas: np.array,
    total_concentration: float,
    ripley_type: str,
):
    x_values = all_distances
    y_values = protein_per_distance / (x_barycentric_areas * total_concentration)

    # cut off infinity values
    non_inf_xvalues = x_values[x_values < np.inf]
    non_inf_yvalues = y_values[: len(non_inf_xvalues)]
    x_values = non_inf_xvalues
    y_values = non_inf_yvalues

    if ripley_type == "L":
        y_values *= np.pi
        y_values *= x_values**2

        y_values = np.sqrt(y_values / np.pi)
        y_values -= x_values
    return x_values, y_values


def aggregate_ripleys_stats(
    ripley_stats: List[dict], ripley_type: str = "L", num_bins: int = 50
):
    assert ripley_type in ["K", "L", "O"]
    # extract relevant arrays
    distance_matrices, mesh_distances, barycentric_areas = get_ripleys_inputs(
        ripley_stats
    )

    avg_starting_points, avg_reachable_points = get_number_of_points(
        distance_matrices=distance_matrices
    )

    # compute distances from all proteins to all vertices
    all_mesh_distances = np.concatenate(
        [np.ravel(mesh_distance) for mesh_distance in mesh_distances]
    )

    # compute barycentric areas for each vertex and stack for each protein
    all_barycentric_areas = get_barycentric_areas(
        barycentric_areas=barycentric_areas, mesh_distances=mesh_distances
    )

    # print(all_barycentric_areas.shape, len(avg_reachable_points), "<--")
    # print(all_mesh_distances.shape, len(avg_reachable_points), "<--")
    # print(avg_reachable_points, avg_starting_points, "<--")

    # # compute global concentration of reachable points
    # total_concentration = np.sum(avg_reachable_points) / np.sum(
    #     all_barycentric_areas[all_mesh_distances < np.inf]
    # )

    # sort in ascending order
    all_mesh_distances, all_barycentric_areas = (
        sort_barycentric_areas_and_mesh_distances(
            all_mesh_distances=all_mesh_distances,
            all_barycentric_areas=all_barycentric_areas,
        )
    )

    # split protein-protein distances into bins
    all_distances, distance_histogram = get_xaxis_distances(
        distance_matrices=distance_matrices, num_bins=num_bins
    )
    protein_per_distance = distance_histogram / avg_starting_points

    # accumulate into computed bins
    protein_per_distance, x_barycentric_areas, total_concentration = (
        accumulate_barycentric_areas(
            all_barycentric_areas=all_barycentric_areas,
            all_mesh_distances=all_mesh_distances,
            all_distances=all_distances,
            protein_per_distance=protein_per_distance,
            ripley_type=ripley_type,
        )
    )

    # define final outputs
    x_values, y_values = define_xy_values(
        all_distances=all_distances,
        protein_per_distance=protein_per_distance,
        x_barycentric_areas=x_barycentric_areas,
        total_concentration=total_concentration,
        ripley_type=ripley_type,
    )

    return x_values, y_values
