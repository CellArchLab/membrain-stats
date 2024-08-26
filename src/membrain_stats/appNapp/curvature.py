import pyvista as pv
from time import time
from scipy.spatial import KDTree, distance_matrix
import numpy as np


def get_membrane_tops(in_mesh, out_mesh_tops):
    time_zero = time()
    print("Computing tops of membranes!")
    pd = pv.read(in_mesh)
    curvature = pd.curvature(curv_type='Minimum') # Minimum principal curvature
    pd.point_data['curv_array_tops'] = curvature
    pd = pd.point_data_to_cell_data()
    unstruc_grid = pd.threshold(-0.2, 'curv_array_tops', invert=True)
    pd_new = unstruc_grid.extract_surface()
    thres_verts = pd_new.points
    # thres_verts = remove_thin_densities(thres_verts, neighbor_thres=5., sum_thres=12)
    orig_verts = pd.points

    query_tree = KDTree(thres_verts)
    min_dists, _ = query_tree.query(orig_verts)
    pd.point_data['min_dists_tops'] = min_dists
    pd = pd.point_data_to_cell_data()
    pd.save(out_mesh_tops)
    print("Getting the curvatures for tops took", time() - time_zero, 'seconds.')


def remove_thin_densities(verts, neighbor_thres=5., sum_thres=4):
    new_verts = []
    for vert in verts:
        neighbor_sum = np.sum(np.linalg.norm(verts - vert, axis=1) < neighbor_thres)
        # print(np.linalg.norm(verts - vert, axis=1).shape)
        # print(neighbor_sum)
        if neighbor_sum > sum_thres:
            new_verts.append(vert)
    new_verts = np.stack(new_verts)
    return new_verts

def get_curvatures(in_mesh, out_mesh_curv):
    time_zero = time()
    pd = pv.read(in_mesh)
    curvature = pd.curvature(curv_type='Maximum') # Maximum principal curvature
    pd.point_data['curv_array'] = curvature
    pd = pd.point_data_to_cell_data()
    unstruc_grid = pd.threshold(0.1, 'curv_array')
    pd_new = unstruc_grid.extract_surface()
    thres_verts = pd_new.points
    thres_verts = remove_thin_densities(thres_verts, neighbor_thres=5., sum_thres=12)
    orig_verts = pd.points

    query_tree = KDTree(thres_verts)
    min_dists, _ = query_tree.query(orig_verts)
    pd.point_data['min_dists'] = min_dists
    pd = pd.point_data_to_cell_data()
    pd.save(out_mesh_curv)
    print("Getting the curvatures took", time() - time_zero, 'seconds.')