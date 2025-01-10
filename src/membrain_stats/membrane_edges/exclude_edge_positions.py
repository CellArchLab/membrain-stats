import os
import numpy as np
import trimesh
import scipy
from membrain_stats.membrane_edges.edge_from_curvature import get_edge_mask


def mask_geodesic_edge_positions(
    positions: np.ndarray,
    mesh_dict: dict,
    edge_exclusion_params: dict,
    return_pos_mask: bool = False,
):
    mesh_orig = trimesh.Trimesh(vertices=mesh_dict["verts"], faces=mesh_dict["faces"])
    mesh = get_edge_mask(
        mesh=mesh_orig,
        edge_exclusion_width=edge_exclusion_params["edge_exclusion_width"],
        temp_file=edge_exclusion_params["out_file"],
        percentile=edge_exclusion_params["edge_percentile"],
        return_vertex_mask=True,
    )
    if edge_exclusion_params["store_sanity_meshes"]:
        # store in file:
        out_mesh_orig = (
            "./sanity_meshes/"
            + os.path.basename(edge_exclusion_params["out_file"]).split(".")[0]
            + "_orig.obj"
        )
        out_mesh_cropped = "./sanity_meshes/" + os.path.basename(
            edge_exclusion_params["out_file"]
        )
        mesh_orig.export(out_mesh_orig)
        mesh[0].export(out_mesh_cropped)

    mesh, vertex_mask = mesh
    positions = mesh_dict["positions"]

    excluded_vertices = mesh_orig.vertices[~vertex_mask]
    included_vertices = mesh_orig.vertices[vertex_mask]

    # compute nearest neighbor vertex for each position
    tree_included = scipy.spatial.cKDTree(included_vertices)
    tree_excluded = scipy.spatial.cKDTree(excluded_vertices)

    nn_distances_inc = tree_included.query(positions, k=1)[0]
    nn_distances_exc = tree_excluded.query(positions, k=1)[0]

    # exclude vertices that are closer to the excluded vertices than to the included vertices
    mask = nn_distances_inc < nn_distances_exc
    if return_pos_mask:
        return mask
    positions = positions[mask]
    return positions, mesh
