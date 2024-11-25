import os
import pandas as pd
import starfile
import trimesh
import scipy

from membrain_stats.utils.io_utils import (
    get_mesh_filenames,
    get_mesh_from_file,
    get_tmp_edge_files,
)
from membrain_stats.membrane_edges.edge_from_curvature import get_edge_mask


def protein_concentration_singlemb(
    filename: str,
    pixel_size_multiplier: float = None,
    pixel_size_multiplier_positions: float = None,
    exclude_edges: bool = False,
    edge_file: str = None,
    edge_exclusion_width: float = 50.0,
    only_one_side: bool = False,
    edge_percentile: float = 95,
    store_sanity_meshes: bool = False,
):
    mesh_dict = get_mesh_from_file(
        filename,
        pixel_size_multiplier=pixel_size_multiplier,
        pixel_size_multiplier_positions=pixel_size_multiplier_positions,
    )
    mesh = trimesh.Trimesh(vertices=mesh_dict["verts"], faces=mesh_dict["faces"])

    if exclude_edges:
        mesh_orig = mesh
        mesh = get_edge_mask(
            mesh=mesh,
            edge_exclusion_width=edge_exclusion_width,
            temp_file=edge_file,
            percentile=edge_percentile,
            return_vertex_mask=True,
        )

        if store_sanity_meshes:
            # store in file:
            out_mesh_orig = (
                "./sanity_meshes/"
                + os.path.basename(filename).split(".")[0]
                + "_orig.obj"
            )
            out_mesh_cropped = "./sanity_meshes/" + os.path.basename(filename)
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
        positions = positions[mask]
        mesh_dict["positions"] = positions

    area = mesh.area
    area = area * (0.5 if only_one_side else 1.0)

    num_proteins = len(mesh_dict["positions"])

    protein_concentration = num_proteins / area
    protein_concentration = protein_concentration * 100  # get value in nm^-2

    return num_proteins, area, protein_concentration


def protein_concentration_folder(
    in_folder: str,
    out_folder: str,
    only_one_side: bool = False,
    exclude_edges: bool = False,
    edge_exclusion_width: float = 50.0,
    pixel_size_multiplier: float = None,
    pixel_size_multiplier_positions: float = None,
    plot: bool = False,
    edge_percentile: float = 95,
    store_sanity_meshes: bool = False,
):

    filenames = get_mesh_filenames(in_folder)
    out_files_edges = get_tmp_edge_files(out_folder, filenames)

    concentration_outputs = [
        protein_concentration_singlemb(
            filename=filename,
            pixel_size_multiplier=pixel_size_multiplier,
            pixel_size_multiplier_positions=pixel_size_multiplier_positions,
            exclude_edges=exclude_edges,
            edge_file=edge_file,
            edge_exclusion_width=edge_exclusion_width,
            only_one_side=only_one_side,
            edge_percentile=edge_percentile,
            store_sanity_meshes=store_sanity_meshes,
        )
        for filename, edge_file in zip(filenames, out_files_edges)
    ]

    num_proteins, areas, protein_concentrations = zip(*concentration_outputs)

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
