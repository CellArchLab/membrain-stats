import os
import numpy as np
import pyvista as pv
from membrain_pick.mesh_projection_utils import get_normals_from_face_order, remove_unused_vertices
from membrain_pick.compute_mesh_projection import convert_seg_to_evenly_spaced_mesh
from membrain_seg.segmentation.dataloading.data_utils import load_tomogram
from membrain_stats.appNapp.normal_tracing import get_distances, get_appNapp_areas
from membrain_stats.appNapp.curvature import get_curvatures, get_membrane_tops



orig_seg_path = "/scicore/home/engel0006/GROUP/pool-engel/Lorenz/MemBrain-seg/publication/test_tomo_predictions/tomo17_bin4_denoised_MemBrain_seg_incremental_training_vtest_v6_run1-epoch=999_segmented.mrc"
out_folder = "/scicore/home/engel0006/GROUP/pool-engel/Lorenz/MemBrain_stats/tests_appnapp"

seg_folder = "/scicore/home/engel0006/GROUP/pool-engel/Spinach_project/Segmented_thylakoids"

single_file = "180524_tomo1_thylakoids.mrc" # None

for filename in os.listdir(seg_folder):
    if single_file is not None and filename != single_file:
        continue
    orig_seg_path = os.path.join(seg_folder, filename)
    out_file_token = os.path.basename(orig_seg_path).replace(".mrc", "")

    print(f"Processing {orig_seg_path}")
    seg = load_tomogram(orig_seg_path).data

    print("Converting to mesh")
    mesh = convert_seg_to_evenly_spaced_mesh(seg=seg,
                    smoothing=2000,
                    input_pixel_size=14.08,
                    output_pixel_size=14.08,
                    barycentric_area=3.)

    points, faces, face_normals = get_normals_from_face_order(mesh, return_face_normals=True)
    points, faces, _ = remove_unused_vertices(points, faces, np.zeros_like(points))

    faces = np.hstack([np.full((faces.shape[0], 1), 3, dtype=int), faces])
    faces = faces.reshape(-1)
    new_mesh = pv.PolyData(points, faces)


    # store the mesh
    mesh_path = os.path.join(out_folder, f"{out_file_token}_mesh_step1.vtp")
    new_mesh.save(mesh_path) 

    print(f"Getting distance")
    # get the distances
    distance_out_mesh = os.path.join(out_folder, f"{out_file_token}_mesh_step2.vtp")
    distances = get_distances(mesh_path, distance_out_mesh, dist_thres=6, use_rotational_normals=False)

    print(f"Getting curvature")
    # get curvature
    curvature_out_mesh = os.path.join(out_folder, f"{out_file_token}_mesh_step3.vtp")
    curvatures = get_curvatures(distance_out_mesh, curvature_out_mesh)

    print(f"Getting membrane tops")
    # get membrane tops
    membrane_tops_out_mesh = os.path.join(out_folder, f"{out_file_token}_mesh_step4.vtp")
    membrane_tops = get_membrane_tops(curvature_out_mesh, membrane_tops_out_mesh)

    print(f"Getting appNapp areas")
    # get appNapp areas
    appNapp_areas_out_csv = os.path.join(out_folder, f"{out_file_token}_mesh_step5.csv")
    appNapp_areas = get_appNapp_areas(appNapp_areas_out_csv, membrane_tops_out_mesh,
                                    exclude_tops=True, tops_thres=2.0, 
                                    divide_stats_by2=False,
    )

