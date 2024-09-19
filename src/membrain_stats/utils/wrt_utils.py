from typing import List, Tuple
import numpy as np
import trimesh
from membrain_stats.utils.mesh_utils import barycentric_area_per_vertex
from membrain_stats.utils.geodesic_distance_utils import (
    compute_geodesic_distance_matrix,
    compute_euclidean_distance_matrix,
)


def extract_positions_by_class(
    mesh_dicts: List[dict], with_respect_to_class: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Extracts positions of the specified class and other classes to consider."""
    protein_classes = [mesh_dict["classes"] for mesh_dict in mesh_dicts]
    wrt_positions = [
        mesh_dict["positions"][mesh_dict["classes"] == with_respect_to_class]
        for mesh_dict in mesh_dicts
    ]
    return protein_classes, wrt_positions


def create_class_masks(
    protein_classes: List[np.ndarray],
    consider_classes: List[int],
    with_respect_to_class: int,
) -> List[np.ndarray]:
    """Creates masks for the classes we want to consider and those for the reference class."""
    if -1 in consider_classes:
        masks = [classes != with_respect_to_class for classes in protein_classes]
        edge_masks = [classes != -1 for classes in protein_classes]
        masks = [mask & edge_mask for mask, edge_mask in zip(masks, edge_masks)]
    else:
        masks = [np.isin(classes, consider_classes) for classes in protein_classes]
    return masks


def get_positions_for_classes(
    mesh_dicts: List[dict], masks: List[np.ndarray]
) -> List[np.ndarray]:
    """Extracts positions for the classes we are considering based on the masks."""
    return [mesh_dict["positions"][mask] for mask, mesh_dict in zip(masks, mesh_dicts)]


def compute_distance_matrices(
    meshes: List[trimesh.Trimesh],
    wrt_positions: List[np.ndarray],
    consider_positions: List[np.ndarray],
    distance_matrix_method: str,
    geod_distance_method: str,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Computes the distance matrices (either geodesic or euclidean) and mesh distances."""
    if distance_matrix_method == "geodesic":
        distance_matrix_outputs = [
            compute_geodesic_distance_matrix(
                verts=mesh.vertices,
                faces=mesh.faces,
                point_coordinates_target=consider_positions[i],
                point_coordinates=wrt_positions[i],
                method=geod_distance_method,
                return_mesh_distances=True,
                infinite_distance_for_pointless_areas=True,
            )
            for i, mesh in enumerate(meshes)
        ]
    elif distance_matrix_method == "euclidean":
        distance_matrix_outputs = [
            compute_euclidean_distance_matrix(
                verts=mesh.vertices,
                point_coordinates_target=consider_positions[i],
                point_coordinates=wrt_positions[i],
                return_mesh_distances=True,
            )
            for i, mesh in enumerate(meshes)
        ]
    distance_matrices = [output[0] for output in distance_matrix_outputs]
    mesh_distances = [output[1] for output in distance_matrix_outputs]

    return distance_matrices, mesh_distances


def compute_barycentric_areas(meshes: List[trimesh.Trimesh]) -> List[np.ndarray]:
    """Computes the barycentric area per vertex for each mesh."""
    return [barycentric_area_per_vertex(mesh) for mesh in meshes]


def flatten_and_concatenate(arrays: List[np.ndarray]) -> np.ndarray:
    """Flattens and concatenates a list of arrays."""
    return np.concatenate([np.ravel(array) for array in arrays], axis=0)


def get_wrt_inputs(
    mesh_dicts: List[dict],
    meshes: List[trimesh.Trimesh],
    consider_classes: List[int],
    with_respect_to_class: int,
    geod_distance_method: str,
    distance_matrix_method: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Main function to compute protein distances and mesh information."""
    # Extract positions based on the classes
    protein_classes, wrt_positions = extract_positions_by_class(
        mesh_dicts, with_respect_to_class
    )

    # Create masks for the classes we are considering
    masks = create_class_masks(protein_classes, consider_classes, with_respect_to_class)

    # Get positions for the classes we are considering
    consider_positions = get_positions_for_classes(mesh_dicts, masks)

    # Compute the distance matrices
    distance_matrices, mesh_distances = compute_distance_matrices(
        meshes,
        wrt_positions,
        consider_positions,
        distance_matrix_method,
        geod_distance_method,
    )
    # Compute the nearest distances for each protein
    protein_nearest_wrt_distances = [
        np.min(distance_matrix, axis=0) for distance_matrix in distance_matrices
    ]
    protein_nearest_wrt_distances = np.concatenate(protein_nearest_wrt_distances)
    protein_nearest_wrt_distances = np.sort(protein_nearest_wrt_distances)

    # Compute the barycentric areas

    mesh_distances = [np.min(mesh_dist, axis=0) for mesh_dist in mesh_distances]

    mesh_barycentric_areas = compute_barycentric_areas(meshes)
    mesh_barycentric_areas = np.concatenate(mesh_barycentric_areas)

    # Flatten and concatenate mesh distances
    mesh_distances = flatten_and_concatenate(mesh_distances)

    return protein_nearest_wrt_distances, mesh_barycentric_areas, mesh_distances
