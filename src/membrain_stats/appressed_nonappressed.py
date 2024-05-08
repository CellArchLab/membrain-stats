import numpy as np
import pyvista as pv
from membrain_pick.mesh_projection_utils import convert_seg_to_evenly_spaced_mesh, get_normals_from_face_order, remove_unused_vertices
from membrain_seg.segmentation.dataloading.data_utils import load_tomogram
from membrain_stats.appNapp.normal_tracing import get_distances, get_appNapp_areas
from membrain_stats.appNapp.curvature import get_curvatures, get_membrane_tops



mesh = convert_seg_to_evenly_spaced_mesh(seg=cur_seg,
                smoothing=mesh_smoothing,
                input_pixel_size=input_pixel_size,
                output_pixel_size=output_pixel_size,
                barycentric_area=barycentric_area)

points, faces, face_normals = get_normals_from_face_order(mesh, return_face_normals=True)
points, faces, _ = remove_unused_vertices(points, faces, np.zeros_like(points))

