import numpy as np
import scipy.spatial

import pyvista as pv
from membrain_pick.mesh_projection_utils import get_normals_from_face_order
from membrain_pick.mesh_class import Mesh
from membrain_pick.dataloading.data_utils import store_array_in_csv

def z_rot_matrix(phi):
    return np.array(np.array([[np.cos(phi), -1* np.sin(phi), 0],[np.sin(phi), np.cos(phi), 0], [0, 0, 1]]))

def y_rot_matrix(psi):
    return np.array([[np.cos(psi), 0, -1 * np.sin(psi)], [0, 1, 0], [np.sin(psi), 0, np.cos(psi)]])

def x_rot_matrix(alpha):
    return np.array([[1, 0, 0],[0, np.cos(alpha), -1 * np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])

def compute_distances_in_varied_normals(pd, use_rotational_normals=False, add_const=7.0):
    curvature = pd.curvature(curv_type='Minimum')
    pd.point_data['curv_array'] = curvature
    pd = pd.point_data_to_cell_data()
    curv_mask = pd.get_array('curv_array') < -0.15


    pd_norms = pd.compute_normals(flip_normals=True, progress_bar=True)
    normals = pd_norms.get_array("Normals")  #TODO: Get some variation into normals
    if use_rotational_normals:
        rotation_matrices = [x_rot_matrix(+0.0 * np.pi),
                            x_rot_matrix(+0.2 * np.pi),
                            x_rot_matrix(-0.2 * np.pi),
                            y_rot_matrix(+0.2 * np.pi),
                            y_rot_matrix(-0.2 * np.pi),
                            z_rot_matrix(+0.2 * np.pi),
                            z_rot_matrix(-0.2 * np.pi)
                            ]
        min_dists = np.zeros((normals.shape[0], 7))
    else:
        rotation_matrices = [x_rot_matrix(+0.0 * np.pi),
                            ]
        min_dists = np.zeros((normals.shape[0], 1))
    for rot_nr, rot_mat in enumerate(rotation_matrices):
        cur_normals = np.dot(rot_mat, normals.T).T
        centers = pd_norms.cell_centers()
        center_coords = centers.points.copy()
        center_coords_orig = centers.points.copy()
        center_coords += (add_const if rot_nr == 0 else (add_const + 1.0)) * cur_normals

        kdtree = scipy.spatial.cKDTree(center_coords_orig)
        cur_min_dists = kdtree.query(center_coords)[0]
        cur_min_dists[curv_mask] = 0.1
        min_dists[:, rot_nr] = cur_min_dists
    min_dists_orig = min_dists[:, 0]
    min_dists = np.min(min_dists, axis=1)
    return min_dists, min_dists_orig

def threshold_and_get_opposed_mb(pd, threshold, argument):
    pd_norms = pd.compute_normals(flip_normals=True, progress_bar=True)
    centers = pd_norms.cell_centers()
    center_coords1 = centers.points.copy()
    center_coords_thres = pd.cell_centers().points.copy()[pd.cell_data[argument] > threshold]
    normals = pd_norms.get_array("Normals")[pd.cell_data[argument] > threshold]
    center_coords_thres -= 3.5*normals
    kdtree = scipy.spatial.cKDTree(center_coords_thres)
    min_dists = kdtree.query(center_coords1)[0]
    pd.cell_data['opposing_distance'] = min_dists
    pd.cell_data['either_argument'] = np.bitwise_or(min_dists < 2.5,pd.cell_data[argument] > threshold) * 1.0
    return pd

def smoothen_argument(pd, argument, nn_k, threshold, below=True):
    center_coords = pd.cell_centers().points.copy()
    center_coords_compare = pd.cell_centers().points.copy()[((1.0 if below else -1.0) * pd.cell_data[argument]) <
                                                            ((1.0 if below else -1.0) * threshold)]
    kdtree = scipy.spatial.cKDTree(center_coords_compare)
    min_dists = kdtree.query(center_coords, k=nn_k)[0]
    avg = np.mean(min_dists, axis=1)
    pd.cell_data[argument + '_smoothed'] = avg
    return pd

def get_distances(in_mesh, out_mesh, dist_thres=6.0, use_rotational_normals=False):
    print("Reading mesh")
    pd = pv.read(in_mesh)
    print("mesh read")

    # centers = pd.cell_centers()
    centers = pd.cell_centers()
    print("Getting cell centers")
    data2 = centers.points.copy()
    print("copying done")

    print("Computing distances with respect to normal vectors (Normal vectors are slightly rotated)")
    min_dists, min_dists_orig = compute_distances_in_varied_normals(pd, use_rotational_normals=use_rotational_normals, add_const=dist_thres+1.)

    pd.cell_data['nearest_mb_dist_min'] = min_dists
    pd.cell_data['nearest_mb_dist_orig'] = min_dists_orig


    print('Thresholding the distances, and adding the backside of the membrane (not annotated so far)')
    pd = threshold_and_get_opposed_mb(pd, dist_thres, 'nearest_mb_dist_min')
    pd.save(out_mesh)

    print("Smoothening side picking. Not sure if this is actually working :S :S")
    pd = smoothen_argument(pd, 'either_argument', nn_k=5, threshold=0.5, below=True)
    pd = smoothen_argument(pd, 'either_argument_smoothed', nn_k=8, threshold=4.5, below=False)


    thres_min_dist_points = centers.points.copy()[min_dists >= 12.]
    kdtree2 = scipy.spatial.cKDTree(thres_min_dist_points)
    dists_within_threses = np.sum(kdtree2.query(thres_min_dist_points, k=5)[0], axis=1)
    print(np.unique(dists_within_threses))

    mask_thres_min_dist_points = dists_within_threses < 100.
    thres_min_dist_points_thres = thres_min_dist_points[mask_thres_min_dist_points]
    kdtree3 = scipy.spatial.cKDTree(thres_min_dist_points_thres)
    clean_min_dists = kdtree3.query(data2)[0]


    pd.cell_data['nearest_mb_dist_to_thres_clean'] = clean_min_dists






    pd.save(out_mesh)


def get_area(coords):
    sides1 = np.linalg.norm(coords[:, 1, :] - coords[:, 0, :], axis=1)
    sides2 = np.linalg.norm(coords[:, 2, :] - coords[:, 1, :], axis=1)
    sides3 = np.linalg.norm(coords[:, 0, :] - coords[:, 2, :], axis=1)
    s_array = (sides1 + sides2 + sides3) / 2
    areas = np.sqrt(s_array * (s_array - sides1) * (s_array - sides2) * (s_array - sides3))
    all_area = np.sum(areas)
    return all_area

def get_cell_data(pd):
    print("Getting cell data. This is extremely inefficient!")
    centers = pd.cell_centers()
    n_cells = centers.points.shape[0]
    all_cell_points = []
    for i in range(n_cells):
        cell_points = pd.cell_points(i)
        all_cell_points.append(cell_points)
    return np.stack(all_cell_points)

def get_cell_data(pd):
    print("Getting cell data. This is extremely inefficient!")
    
    # Initialize a cell array to store point coordinates for each cell
    all_cell_points = []

    # Loop over each cell in the PolyData
    for i in range(pd.GetNumberOfCells()):
        cell = pd.GetCell(i)
        points = cell.GetPoints()

        # Extract the point coordinates for this cell
        cell_points = []
        for j in range(points.GetNumberOfPoints()):
            point = [0.0, 0.0, 0.0]
            points.GetPoint(j, point)
            cell_points.append(point)

        # Add this cell's points to the list
        all_cell_points.append(cell_points)

    # Convert to a numpy array for easier handling later on
    return np.array(all_cell_points)


def get_appNapp_areas(out_mesh_curv_area_csv, out_mesh_curv_tops, exclude_tops=False, tops_thres=0.0, out_mesh_curv_area_sanity_mesh1=None, out_mesh_curv_area_sanity_mesh2=None, divide_stats_by2=False):
    print("Computing areas of appressed vs non-appressed regions!")
    pd = pv.read(out_mesh_curv_tops)

    appNapp = pd.cell_data['either_argument']
    app_mask = appNapp < 0.5
    napp_mask = appNapp >= 0.5
    if exclude_tops:
        min_dist_tops = pd.cell_data['min_dists_tops']
        tops_mask = min_dist_tops > tops_thres
        app_mask = np.logical_and(app_mask, tops_mask)
        napp_mask = np.logical_and(napp_mask, tops_mask)

    cell_coords = get_cell_data(pd) #(1544527, 3, 3)
    app_coords = cell_coords[app_mask]
    napp_coords = cell_coords[napp_mask]

    if out_mesh_curv_area_sanity_mesh1 is not None:
        store_as_mesh(out_mesh_curv_area_sanity_mesh1, app_coords)
        store_as_mesh(out_mesh_curv_area_sanity_mesh2, napp_coords)

    app_area = get_area(app_coords)
    napp_area = get_area(napp_coords)

    if divide_stats_by2:
        app_area /= 2
        napp_area /= 2

    print(app_area)
    print(napp_area)
    csv_data = np.expand_dims(np.array((app_area, napp_area)), 0)
    header = ['appressedArea', 'nonAppressedArea']
    store_array_in_csv(out_mesh_curv_area_csv, csv_data)


def store_as_mesh(out_file, verts):
    print("Getting verts and ids for the current thresholded mesh.!")
    ids = []
    points = []
    for vert in verts:
        cur_ids = []
        for i in range(3):
            cur_vert = np.array(vert[i])
            try:
                id = points.index(cur_vert)
                cur_ids.append(id)
            except ValueError:
                points.append(cur_vert)
                cur_ids.append(len(points))
            # print(id)
            # if cur_vert not in points:
            #     points.append(vert)
            #     cur_ids.append(len(points))
            # else:
            #     id = points.index(cur_vert)
            #     cur_ids.append(id)
        cur_ids = np.array(cur_ids)
        ids.append(cur_ids)
    ids = np.stack(ids)
    points = np.stack(points)
    mesh = Mesh(points, ids)
    mesh.store_in_file(out_file)