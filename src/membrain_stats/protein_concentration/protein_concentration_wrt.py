from typing import List
import os
import numpy as np
import pandas as pd
import starfile
import trimesh

from membrain_stats.utils.io_utils import (
    get_mesh_filenames,
    get_mesh_from_file,
)
from membrain_stats.membrane_edges.edge_from_curvature import (
    exclude_edges_from_mesh,
)
from membrain_stats.utils.wrt_utils import get_wrt_inputs


def protein_concentration_wrt_folder(
    in_folder: str,
    out_folder: str,
    exclude_edges: bool = False,
    edge_exclusion_width: float = 50.0,
    pixel_size_multiplier: float = None,
    only_one_side: bool = False,
    consider_classes: List[int] = "all",
    with_respect_to_class: int = 0,
    num_bins: int = 25,
    geod_distance_method: str = "exact",
    distance_matrix_method: str = "geodesic",
):

    filenames = get_mesh_filenames(in_folder)
    mesh_dicts = [
        get_mesh_from_file(filename, pixel_size_multiplier=pixel_size_multiplier)
        for filename in filenames
    ]
    if exclude_edges:
        mesh_dicts = [
            exclude_edges_from_mesh(
                out_folder=out_folder,
                filename=filename,
                mesh_dict=mesh_dict,
                edge_exclusion_width=edge_exclusion_width,
                leave_classes=[with_respect_to_class],
            )
            for filename, mesh_dict in zip(filenames, mesh_dicts)
        ]

    meshes = [
        trimesh.Trimesh(
            vertices=mesh_dict["verts"],
            faces=mesh_dict["faces"],
        )
        for mesh_dict in mesh_dicts
    ]

    protein_nearest_wrt_distances, mesh_barycentric_areas, mesh_distances = (
        get_wrt_inputs(
            mesh_dicts=mesh_dicts,
            meshes=meshes,
            consider_classes=consider_classes,
            with_respect_to_class=with_respect_to_class,
            geod_distance_method=geod_distance_method,
            distance_matrix_method=distance_matrix_method,
        )
    )
    mesh_barycentric_areas /= 100  # convert to nm^2
    if only_one_side:
        mesh_barycentric_areas /= 2

    bins = np.linspace(0, np.max(protein_nearest_wrt_distances), num_bins)
    hist = np.histogram(protein_nearest_wrt_distances, bins=bins)[0]
    y_data = []
    for num_bin, protein_numbers in enumerate(hist):
        bin_lower = bins[num_bin]
        bin_upper = bins[num_bin + 1]
        area_mask = (mesh_distances >= bin_lower) & (mesh_distances < bin_upper)
        y_data.append(protein_numbers / np.sum(mesh_barycentric_areas[area_mask]))
    from matplotlib import pyplot as plt

    plt.figure()
    plt.plot(bins[:-1], y_data)
    plt.xlabel("Distance to nearest protein")
    plt.ylabel("Area covered")
    plt.savefig("./protein_concentration_wrt.png")

    out_data = {
        "bin_lower": bins[:-1],
        "bin_upper": bins[1:],
        "concentration": y_data,
    }
    out_data = pd.DataFrame(out_data)
    starfile.write(
        out_data,
        os.path.join(out_folder, "protein_concentration_wrt.star"),
        float_format="%.12f",
    )
