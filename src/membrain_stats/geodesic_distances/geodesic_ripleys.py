from typing import List
import os
import pandas as pd
import numpy as np
import trimesh
import starfile
from membrain_stats.utils.io_utils import get_mesh_filenames, get_mesh_from_file, get_tmp_edge_files, get_geodesic_distance_input
from membrain_stats.utils.mesh_utils import find_closest_vertices, barycentric_area_per_vertex
from membrain_stats.geodesic_distances.geodesic_distances import compute_geodesic_distance_matrix
from membrain_stats.membrane_edges.edge_from_curvature import get_edge_mask


def compute_ripleys_stats(
        mesh_dict: List[dict],
        method: str = "fast",
):
    distance_matrix, mesh_distances = \
        compute_geodesic_distance_matrix(
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

def get_ripleys_inputs(
        ripley_stats: List[dict]
):
    distance_matrices = [ripley_stat[0] for ripley_stat in ripley_stats]
    mesh_distances = [ripley_stat[1] for ripley_stat in ripley_stats]
    barycentric_areas = [ripley_stat[2] for ripley_stat in ripley_stats]

    return distance_matrices, mesh_distances, barycentric_areas
    
def get_xaxis_distances(
        distance_matrices: List[np.array],
        num_bins: int
):
    # flatten protein-protein distances
    all_distances = np.concatenate([np.ravel(distance_matrix) for distance_matrix in distance_matrices])
    all_distances = all_distances[all_distances != -1] # exlude distance to self

    # sort protein-protein distances
    sort_indices = np.argsort(all_distances)
    all_distances = all_distances[sort_indices]

    # split distances into bins
    distance_histogram, bin_edges = np.histogram(all_distances[all_distances < np.inf], bins=num_bins)
    all_distances = bin_edges[:-1]
    
    return all_distances, distance_histogram

def get_number_of_points(
        distance_matrices: List[np.array]
):
    # compute number of starting and reachable points
    num_starting_points = sum([len(distance_matrix) for distance_matrix in distance_matrices])
    avg_starting_points = num_starting_points / len(distance_matrices)

    num_reachable_points = [[np.sum(distance_matrix[:, i] < np.inf)for i in range(len(distance_matrix))]  for distance_matrix in distance_matrices ]
    avg_reachable_points = [np.mean(entry) for entry in num_reachable_points]

    return avg_starting_points, avg_reachable_points

def get_barycentric_areas(
        barycentric_areas: List[np.array],
        mesh_distances: List[np.array]
):
    # compute barycentric areas for each vertex and shape corresponding to all_mesh_distances
    repeated_barycentric_areas = [
        np.repeat(barycentric_area, len(mesh_distance)) for barycentric_area, mesh_distance in zip(barycentric_areas, mesh_distances)
        ]
    all_barycentric_areas = np.concatenate(
        [np.ravel(repeated_barycentric_area) for repeated_barycentric_area in repeated_barycentric_areas]
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
        ripley_type: str
):
    # compute barycentric areas for each x-axis split (i.e. all_distances)
    x_barycentric_areas = np.split(all_barycentric_areas, np.searchsorted(all_mesh_distances, all_distances))[:-1]
    x_barycentric_areas = np.array([np.sum(x_barycentric_area) for x_barycentric_area in x_barycentric_areas])
    
    # accumulate if not computing O statistic
    if ripley_type != "O":
        protein_per_distance = np.cumsum(protein_per_distance)
        x_barycentric_areas = np.cumsum(x_barycentric_areas) 

    return protein_per_distance, x_barycentric_areas

def define_xy_values(
        all_distances: np.array,
        protein_per_distance: np.array,
        x_barycentric_areas: np.array,
        total_concentration: float,
        ripley_type: str
):
    x_values = all_distances
    y_values = protein_per_distance / (x_barycentric_areas * total_concentration)

    # cut off infinity values
    non_inf_xvalues = x_values[x_values < np.inf]
    non_inf_yvalues = y_values[:len(non_inf_xvalues)]
    x_values = non_inf_xvalues
    y_values = non_inf_yvalues

    if ripley_type == "L":
        y_values *= np.pi 
        y_values *= x_values ** 2

        y_values = np.sqrt(y_values / np.pi)
        y_values -= x_values
    return x_values, y_values


def aggregate_ripleys_stats(
        ripley_stats: List[dict],
        ripley_type: str = "L",
        num_bins: int = 50
):
    assert ripley_type in ["K", "L", "O"]
    # extract relevant arrays
    distance_matrices, mesh_distances, barycentric_areas = get_ripleys_inputs(ripley_stats)

    avg_starting_points, avg_reachable_points = get_number_of_points(
        distance_matrices=distance_matrices
    )

    # compute distances from all proteins to all vertices
    all_mesh_distances = np.concatenate(
        [np.ravel(mesh_distance) for mesh_distance in mesh_distances]
        )

    # compute barycentric areas for each vertex and stack for each protein
    all_barycentric_areas = get_barycentric_areas(
        barycentric_areas=barycentric_areas,
        mesh_distances=mesh_distances
    )

    # compute global concentration of reachable points
    total_concentration = np.sum(avg_reachable_points) / np.sum(all_barycentric_areas[all_mesh_distances < np.inf])
    
    # sort in ascending order
    all_mesh_distances, all_barycentric_areas = sort_barycentric_areas_and_mesh_distances(
        all_mesh_distances=all_mesh_distances,
        all_barycentric_areas=all_barycentric_areas
    )

    # split protein-protein distances into bins
    all_distances, distance_histogram = get_xaxis_distances(
        distance_matrices=distance_matrices,
        num_bins=num_bins
    )
    protein_per_distance = distance_histogram / avg_starting_points

    # accumulate into computed bins
    protein_per_distance, x_barycentric_areas = accumulate_barycentric_areas(
        all_barycentric_areas=all_barycentric_areas,
        all_mesh_distances=all_mesh_distances,
        all_distances=all_distances
    )

    # define final outputs
    x_values, y_values = define_xy_values(
        all_distances=all_distances,
        protein_per_distance=protein_per_distance,
        x_barycentric_areas=x_barycentric_areas,
        total_concentration=total_concentration
    )
    
    return x_values, y_values


def filter_edge_positions(
    positions: np.ndarray,
    all_vertices: np.ndarray,
    edge_vertex_mask: np.ndarray,
):
    """
    Create a mask for positions that are not on the edge.

    Being on the edge is defined as having the nearest vertex of the mesh
    as a vertex of the edge.

    Parameters
    ----------
    positions : np.ndarray
        The positions to filter.    
    all_vertices : np.ndarray
        The vertices of the mesh.
    edge_vertex_mask : np.ndarray
        A boolean mask for the vertices of the mesh that are on the edge.
    """
    closest_vertex_idcs = find_closest_vertices(
        verts=all_vertices, 
        points=positions,
        )
    filter_mask = np.logical_not(edge_vertex_mask[closest_vertex_idcs])
    return filter_mask

def exclude_edges_from_mesh(
        out_folder,
        filename,
        mesh_dict,
        edge_exclusion_width,
        ):
    # get temp_filename (will be created if non-existent)
    out_file_edges = get_tmp_edge_files(out_folder, [filename])[0]

    # initialize mesh
    mesh = trimesh.Trimesh(
            vertices=mesh_dict["verts"], 
            faces=mesh_dict["faces"]
        ) 
    orig_verts = mesh_dict["verts"].copy()
        
    # get the mesh edge mask (entry 0 is masked mesh, entry 1 is mask)
    mesh = get_edge_mask(
                mesh=mesh, 
                edge_exclusion_width=edge_exclusion_width, 
                temp_file=out_file_edges,
                return_vertex_mask=True,
                )
    edge_vertex_mask = mesh[1]
    mesh = mesh[0]

    # set new vertices
    mesh_dict["verts"] = mesh.vertices
    mesh_dict["faces"] = mesh.faces

    # mask out edge positions (i.e. position with nearest neighbor on the excluded edge)
    pos_filter_mask = filter_edge_positions(mesh_dict["positions"], orig_verts, edge_vertex_mask)
    mesh_dict["classes"][pos_filter_mask] = -1
    return mesh_dict


def geodesic_ripleys_folder(
        in_folder: str,
        out_folder: str,
        pixel_size_multiplier: float = None,
        start_classes: List[int] = [0],
        target_classes: List[int] = [0],
        ripley_type: str = "O",
        num_bins: int = 100,
        method: str = "fast",
        exclude_edges: bool = False,
        edge_exclusion_width: float = 50.,
):
    # get filenames from folder
    filenames = get_mesh_filenames(in_folder)
    # filenames = filenames[1:4]

    # load mehes
    mesh_dicts = [get_mesh_from_file(filename, pixel_size_multiplier=pixel_size_multiplier) for filename in filenames]

    # exclude edges
    if exclude_edges:
        mesh_dicts = [
            exclude_edges_from_mesh(
                out_folder=out_folder,
                filename=filename,
                mesh_dict=mesh_dict,
                edge_exclusion_width=edge_exclusion_width
            ) for (filename, mesh_dict) in zip(filenames, mesh_dicts)
        ]

    # prepare input for computation of geodesic distances
    mesh_dicts = [get_geodesic_distance_input(mesh_dict, start_classes, target_classes) for mesh_dict in mesh_dicts]

    # compute values necessary for ripley's statistics
    ripley_stats = [compute_ripleys_stats(mesh_dict, method=method) for mesh_dict in mesh_dicts]

    # aggregate computed values to output global ripley's statistics
    ripley_stats = \
        aggregate_ripleys_stats(
            ripley_stats=ripley_stats,
            ripley_type=ripley_type,
            num_bins=num_bins
        )
    
    # store in star file
    out_data = {
        "ripleyType": ripley_type,
        "x_values": ripley_stats[0],
        "y_values": ripley_stats[1],
    }
    out_data = pd.DataFrame(out_data)
    out_file = os.path.join(out_folder, f"ripleys{ripley_type}.star")
    os.makedirs(out_folder, exist_ok=True)
    starfile.write(out_data, out_file)