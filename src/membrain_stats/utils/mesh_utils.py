import numpy as np
import trimesh
from trimesh.graph import connected_components, face_adjacency


def resort_mesh(
        verts: np.ndarray, 
        faces: np.ndarray,
        return_mapping: bool = False,
        ):
    """ Resort the mesh so that the vertices are numbered from 0 to n-1.
    
    Inputs:
    - verts (np.ndarray): The vertices of the mesh.
    - faces (np.ndarray): The faces of the mesh. 

    Note: Not all vertices are used in the faces, so we need to find the used vertices and renumber them.
    """
    used_verts = np.unique(faces)
    # create a mapping from the old vertex indices to the new ones
    mapping = {old_index: new_index for new_index, old_index in enumerate(used_verts)}
    # create the new vertices
    new_verts = verts[used_verts]
    # create the new faces
    new_faces = np.array([[mapping[old_index] for old_index in face] for face in faces])
    if return_mapping:
        return new_verts, new_faces, mapping
    return new_verts, new_faces


def find_closest_vertices(verts, points):
    """Find the index of the closest vertex to a given point."""
    if len(points.shape) == 1:
        points = points[np.newaxis, :]
    distances = np.linalg.norm(verts[:, np.newaxis] - points, axis=2)
    return np.argmin(distances, axis=0)

def barycentric_area_per_vertex(mesh: trimesh.Trimesh):
    """ Compute the barycentric area per vertex of a mesh."""
    areas = np.zeros(len(mesh.vertices))
    # add up triangle areas that contain the vertex
    for face in mesh.faces:
        v0, v1, v2 = mesh.vertices[face]
        a = np.linalg.norm(v1 - v2)
        b = np.linalg.norm(v0 - v2)
        c = np.linalg.norm(v0 - v1)
        s = (a + b + c) / 2
        areas[face] += np.sqrt(s * (s - a) * (s - b) * (s - c))
    # divide by 3 to get the barycentric area per vertex
    # (each quadrilateral has equal area at each vertex)
    areas /= 3
    return areas


def split_mesh_into_connected_components(verts, faces, return_face_mapping=False, return_vertex_mapping=False):
    adjacency = face_adjacency(faces)
    components = connected_components(adjacency)

    component_face_idcs = [np.argwhere(np.isin(np.arange(len(faces)), component)).flatten() for component in components]
    component_faces = [faces[component_face_idx] for component_face_idx in component_face_idcs]

    forward_face_mapping = {
        face_idx: (component_idx, component_face_idx) for component_idx, cur_component_face_idcs in enumerate(component_face_idcs) for component_face_idx, face_idx in enumerate(cur_component_face_idcs)
    }

    reverse_face_mapping = {
        (component_idx, component_face_idx): face_idx for face_idx, (component_idx, component_face_idx) in forward_face_mapping.items()
    }
    
    resorted_meshes = [resort_mesh(verts, component_face, return_mapping=True) for component_face in component_faces]
    component_verts = [resorted_mesh[0] for resorted_mesh in resorted_meshes]
    component_faces = [resorted_mesh[1] for resorted_mesh in resorted_meshes]
    vertex_mappings = [resorted_mesh[2] for resorted_mesh in resorted_meshes]

    # Forward vertex mapping: original index -> (component_idx, component_vertex_idx)
    forward_vertex_mapping = {
        vertex_idx: (component_idx, cur_vertex_mapping[vertex_idx]) for component_idx, cur_vertex_mapping in enumerate(vertex_mappings) for vertex_idx in cur_vertex_mapping
    }
    
    # Reverse vertex mapping: (component_idx, component_vertex_idx) -> original index
    reverse_vertex_mapping = {
        (component_idx, component_vertex_idx): vertex_idx for vertex_idx, (component_idx, component_vertex_idx) in forward_vertex_mapping.items()
    }

    out = (component_verts, component_faces)
    if return_face_mapping:
        out += (forward_face_mapping, reverse_face_mapping,)
    if return_vertex_mapping:
        out += (forward_vertex_mapping, reverse_vertex_mapping,)
    return out


