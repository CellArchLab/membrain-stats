import os
import pandas as pd
import starfile
import trimesh

from membrain_stats.utils.io_utils import (
    get_mesh_filenames,
    get_mesh_from_file,
    get_tmp_edge_files,
)
from membrain_stats.membrane_edges.edge_from_curvature import get_edge_mask


def protein_concentration_folder(
    in_folder: str,
    out_folder: str,
    only_one_side: bool = False,
    exclude_edges: bool = False,
    edge_exclusion_width: float = 50.0,
    pixel_size_multiplier: float = None,
):

    filenames = get_mesh_filenames(in_folder)
    mesh_dicts = [get_mesh_from_file(filename) for filename in filenames]
    meshes = [
        trimesh.Trimesh(
            vertices=mesh_dict["verts"]
            * (1.0 if pixel_size_multiplier is None else pixel_size_multiplier),
            faces=mesh_dict["faces"],
        )
        for mesh_dict in mesh_dicts
    ]

    if exclude_edges:
        out_files_edges = get_tmp_edge_files(out_folder, filenames)
        print("Excluding edges. This can take a while.")
        meshes = [
            get_edge_mask(
                mesh=mesh, edge_exclusion_width=edge_exclusion_width, temp_file=filename
            )[0]
            for mesh, filename in zip(meshes, out_files_edges)
        ]

    areas = [mesh.area for mesh in meshes]
    areas = [area * (0.5 if only_one_side else 1.0) for area in areas]

    num_proteins = [len(mesh_dict["positions"]) for mesh_dict in mesh_dicts]

    protein_concentrations = [num_proteins[i] / areas[i] for i in range(len(meshes))]
    protein_concentrations = [
        concentration * 100 for concentration in protein_concentrations
    ]  # get value in nm^-2

    out_data = {
        "filename": filenames,
        "num_proteins": num_proteins,
        "area": areas,
        "protein_concentration": protein_concentrations,
    }
    out_data = pd.DataFrame(out_data)
    out_file = os.path.join(out_folder, "protein_concentration.star")
    os.makedirs(out_folder, exist_ok=True)
    starfile.write(out_data, out_file)
