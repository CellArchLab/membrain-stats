import os
import numpy as np
import trimesh
import pyvista as pv
from scipy.spatial import cKDTree
from membrain_stats.utils.mesh_utils import resort_mesh, find_closest_vertices
from membrain_stats.utils.io_utils import get_tmp_edge_files


def get_edge_mask(
    mesh: trimesh.Trimesh,
    edge_exclusion_width: float,
    percentile: float = 95,
    return_triangle_mask: bool = False,
    return_vertex_mask: bool = False,
    temp_file: str = None,
    force_recompute: bool = False,
):
    """Find edges via high curvature regions and otsu thresholding."""
    curvature = None
    if temp_file is not None and not force_recompute:
        if os.path.exists(temp_file):
            curvature = np.load(temp_file)

    if curvature is None:
        # Get the curvature of the mesh
        print("Computing curvature...")

        pv_mesh = pv.wrap(mesh)
        curvature = pv_mesh.curvature("Mean")
        pv_mesh["Curvature"] = curvature

        if temp_file is not None:
            np.save(temp_file, curvature)

    # get the mask of high curvature regions
    mask = np.abs(curvature) > np.percentile(np.abs(curvature), percentile)
    mask_points = mesh.vertices[mask]

    # get distances to nearest neighbors
    tree = cKDTree(mask_points)
    distances, _ = tree.query(mesh.vertices, k=1)
    distance_mask = distances > edge_exclusion_width

    # find triangles with all vertices in the mask
    triangles = mesh.faces
    triangle_indices = np.arange(len(triangles))
    triangle_mask = distance_mask[triangles].all(axis=1)
    triangle_indices = triangle_indices[triangle_mask]
    new_triangles = triangles[triangle_indices]

    # resort the mesh
    new_verts, new_faces = resort_mesh(mesh.vertices, new_triangles)
    new_mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces)

    out = (new_mesh,)
    if return_triangle_mask:
        out += (triangle_mask,)
    if return_vertex_mask:
        out += (distance_mask,)
    return out


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
    leave_classes=None,
    force_recompute=False,
    percentile=95,
):
    print("Excluding edges from mesh: ", filename)
    # get temp_filename (will be created if non-existent)
    out_file_edges = get_tmp_edge_files(out_folder, [filename])[0]

    # initialize mesh
    mesh = trimesh.Trimesh(vertices=mesh_dict["verts"], faces=mesh_dict["faces"])
    orig_verts = mesh_dict["verts"].copy()

    # get the mesh edge mask (entry 0 is masked mesh, entry 1 is mask)
    mesh = get_edge_mask(
        mesh=mesh,
        edge_exclusion_width=edge_exclusion_width,
        temp_file=out_file_edges,
        return_vertex_mask=True,
        force_recompute=force_recompute,
        percentile=percentile,
    )
    edge_vertex_mask = mesh[1]
    mesh = mesh[0]

    # set new vertices
    mesh_dict["verts"] = mesh.vertices
    mesh_dict["faces"] = mesh.faces

    # mask out edge positions (i.e. position with nearest neighbor on the excluded edge)
    pos_filter_mask = filter_edge_positions(
        mesh_dict["positions"], orig_verts, edge_vertex_mask
    )
    if leave_classes is not None:
        leave_mask = np.isin(mesh_dict["classes"], leave_classes)
        pos_filter_mask = np.logical_and(pos_filter_mask, np.logical_not(leave_mask))
    mesh_dict["classes"][pos_filter_mask] = -1
    return mesh_dict
