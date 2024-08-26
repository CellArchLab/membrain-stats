import os
import trimesh
import starfile
from membrain_pick.dataloading.data_utils import load_mesh_from_hdf5

def get_mesh_filenames(in_folder: str):
    h5_files = [filename for filename in os.listdir(in_folder) if filename.endswith(".h5")]
    obj_files = [filename for filename in os.listdir(in_folder) if filename.endswith(".obj")]

    if len(h5_files) >= len(obj_files):
        files = h5_files
    else:
        files = obj_files

    files = [os.path.join(in_folder, filename) for filename in files]
    files = sorted(files)
    return files

def get_mesh_from_file(filename: str):
    if filename.endswith(".h5"):
        mesh_data = load_mesh_from_hdf5(filename)
        verts = mesh_data["points"]
        faces = mesh_data["faces"]
        positions = mesh_data["cluster_centers"]
    else:
        mesh = trimesh.load_mesh(filename)
        verts = mesh.vertices
        faces = mesh.faces
        pos_file = filename.replace(".obj", "_clusters.star")
        positions = starfile.read(pos_file)
        positions = positions[["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]].values
    out_dict = {
        "verts": verts,
        "faces": faces,
        "positions": positions,
    }
    return out_dict

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
            os.path.basename(filename).replace(".h5", f"_{h5_token}_edges.npy").replace(".obj", f"_{h5_token}_edges.npy")
            ) for filename, h5_token in zip(filenames, h5_tokens)
            ]
    return out_files