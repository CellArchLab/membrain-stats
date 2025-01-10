import os
import numpy as np
import trimesh
import starfile
from membrain_pick.dataloading.data_utils import load_mesh_from_hdf5


def get_mesh_filenames(in_folder: str):
    h5_files = [
        filename for filename in os.listdir(in_folder) if filename.endswith(".h5")
    ]
    obj_files = [
        filename for filename in os.listdir(in_folder) if filename.endswith(".obj")
    ]
    if len(h5_files) >= len(obj_files):
        files = h5_files
    else:
        files = obj_files

    files = [os.path.join(in_folder, filename) for filename in files]
    files = sorted(files)
    return files


def get_mesh_from_file(
    filename: str,
    pixel_size_multiplier: float = None,
    pixel_size_multiplier_positions: float = None,
):
    pixel_size_multiplier = (
        1.0 if pixel_size_multiplier is None else pixel_size_multiplier
    )
    pixel_size_multiplier_positions = (
        1.0
        if pixel_size_multiplier_positions is None
        else pixel_size_multiplier_positions
    )
    if filename.endswith(".h5"):
        mesh_data = load_mesh_from_hdf5(filename)
        verts = mesh_data["points"] * pixel_size_multiplier
        faces = mesh_data["faces"]
        positions = mesh_data["cluster_centers"] * pixel_size_multiplier_positions
        classes = np.zeros(len(positions), dtype=int)
        angles = np.zeros((len(positions), 3))
        hasAngles = False
    else:
        mesh = trimesh.load_mesh(filename)
        verts = mesh.vertices * pixel_size_multiplier
        faces = mesh.faces
        pos_file = filename.replace(".obj", "_clusters.star")
        positions = starfile.read(pos_file)
        if "rlnClassNumber" in positions.columns:
            classes = positions["rlnClassNumber"].values
        else:
            classes = np.zeros(len(positions), dtype=int)
        if "rlnAngleRot" in positions.columns:
            angles = positions[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].values
            hasAngles = True
        else:
            angles = np.zeros((len(positions), 3))
            hasAngles = False
        positions = (
            positions[["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]].values
            * pixel_size_multiplier_positions
        )
    out_dict = {
        "verts": verts,
        "faces": faces,
        "positions": positions,
        "classes": classes,
        "angles": angles,
        "hasAngles": hasAngles,
    }
    return out_dict


def get_geodesic_distance_input(
    mesh_dict: dict,
    start_classes: list,
    target_classes: list,
):
    """Get the input for the geodesic distance computation."""
    classes = mesh_dict["classes"]
    class_start_mask = np.isin(classes, start_classes)
    class_target_mask = np.isin(classes, target_classes)

    positions_start = mesh_dict["positions"][class_start_mask]
    positions_target = mesh_dict["positions"][class_target_mask]

    angles_start = mesh_dict["angles"][class_start_mask]
    angles_target = mesh_dict["angles"][class_target_mask]

    mesh_dict["positions_start"] = positions_start
    mesh_dict["positions_target"] = positions_target
    mesh_dict["angles_start"] = angles_start
    mesh_dict["angles_target"] = angles_target

    return mesh_dict


def get_tmp_edge_files(
    out_folder: str,
    filenames: list,
):
    """
    Get temporary edge files for each mesh file.

    This is trying to store the meshes in a default location, and then return the filenames of the edge files.
    """
    out_folder = os.path.join(os.path.dirname(out_folder), "mesh_edges")
    h5_tokens = ["h5" if filename.endswith(".h5") else "obj" for filename in filenames]
    os.makedirs(out_folder, exist_ok=True)
    out_files = [
        os.path.join(
            out_folder,
            os.path.basename(filename)
            .replace(".h5", f"_{h5_token}_edges.npy")
            .replace(".obj", f"_{h5_token}_edges.npy"),
        )
        for filename, h5_token in zip(filenames, h5_tokens)
    ]
    return out_files
