import os
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from membrain_stats.utils.mesh_utils import resort_mesh

def get_edge_mask(
        mesh: trimesh.Trimesh, 
        edge_exclusion_width: float, 
        percentile: float = 95,
        return_triangle_mask: bool = False,
        return_vertex_mask: bool = False,
        temp_file: str = None,
        ):
    """ Find edges via high curvature regions and otsu thresholding."""
    curvature = None
    if temp_file is not None:
        if os.path.exists(temp_file):
            curvature = np.load(temp_file)
    
    if curvature is None:
        # Get the curvature of the mesh
        curvature = trimesh.curvature.discrete_mean_curvature_measure(mesh, points=mesh.vertices, radius=10.)
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

    # # store mesh in temporary file
    # new_mesh.export("temp_edge_mesh.obj")
    # exit()

    out = (new_mesh, )
    if return_triangle_mask:
        out += (triangle_mask, )
    if return_vertex_mask:
        out += (distance_mask, )
    return out
