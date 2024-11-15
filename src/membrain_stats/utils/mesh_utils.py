from collections import defaultdict
from typing import List
import numpy as np
import trimesh
from trimesh.graph import connected_components, face_adjacency


def resort_mesh(
    verts: np.ndarray,
    faces: np.ndarray,
    return_mapping: bool = False,
):
    """Resort the mesh so that the vertices are numbered from 0 to n-1.

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
    """Compute the barycentric area per vertex of a mesh."""
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


def check_mappings(
    orig_verts: np.ndarray,
    orig_faces: np.ndarray,
    component_verts: List[np.ndarray],
    component_faces: List[np.ndarray],
    forward_vertex_mapping: dict,
    reverse_vertex_mapping: dict,
    forward_face_mapping: dict,
    reverse_face_mapping: dict,
):
    """Check that the mappings are correct.
    Supposed behaviors:
    - forward_vertex_mapping[vertex_idx] = (component_idx, component_vertex_idx)
    - reverse_vertex_mapping[(component_idx, component_vertex_idx)] = vertex_idx
    - forward_face_mapping[face_idx] = (component_idx, component_face_idx)
    - reverse_face_mapping[(component_idx, component_face_idx)] = face_idx
    These mappings should be bijections.
    """
    # check forward mappings defined for all vertices
    assert len(forward_vertex_mapping) == len(orig_verts), (
        "Forward vertex mapping failed because: "
        + str(len(forward_vertex_mapping))
        + " != "
        + str(len(orig_verts))
    )
    # check reverse mappings defined for all forward mappings
    assert len(reverse_vertex_mapping) == len(forward_vertex_mapping)
    # check forward mappings defined for all faces
    assert len(forward_face_mapping) == len(orig_faces)
    # check reverse mappings defined for all forward mappings
    assert len(reverse_face_mapping) == len(forward_face_mapping)
    # check that the forward mappings are bijections
    assert len(set(forward_vertex_mapping.values())) == len(forward_vertex_mapping)
    assert len(set(forward_face_mapping.values())) == len(forward_face_mapping)
    # check that the reverse mappings are bijections
    assert len(set(reverse_vertex_mapping.values())) == len(reverse_vertex_mapping)
    assert len(set(reverse_face_mapping.values())) == len(reverse_face_mapping)
    # check that the forward and reverse mappings are inverses
    for vertex_idx, (
        component_idx,
        component_vertex_idx,
    ) in forward_vertex_mapping.items():
        assert (
            reverse_vertex_mapping[(component_idx, component_vertex_idx)] == vertex_idx
        )
    for face_idx, (component_idx, component_face_idx) in forward_face_mapping.items():
        assert reverse_face_mapping[(component_idx, component_face_idx)] == face_idx
    # check that the vertices and faces are correctly mapped
    for component_idx, (component_verts, component_faces) in enumerate(
        zip(component_verts, component_faces)
    ):
        for component_vertex_idx, component_vertex in enumerate(component_verts):
            vertex_idx = reverse_vertex_mapping[(component_idx, component_vertex_idx)]
            assert np.allclose(orig_verts[vertex_idx], component_vertex)


def vertex_adjacency(faces):
    """
    Returns an (n, 2) list of face indices.
    Each pair of faces in the list shares a vertex, making them adjacent.
    """
    adjacency = []
    for face_idx, face in enumerate(faces):
        for vertex in face:
            adjacent_faces = np.argwhere(np.isin(faces, vertex)).flatten()
            adjacency.extend(
                [(face_idx, adj_face_idx) for adj_face_idx in adjacent_faces]
            )
    return np.array(adjacency)


def vertex_adjacency(faces):
    """
    Returns an (n, 2) array of face indices.
    Each pair of faces in the list shares a vertex, making them adjacent.
    """
    # Dictionary to map each vertex to the faces that contain it
    vertex_to_faces = defaultdict(list)

    # Step 1: Build the vertex-to-face mapping
    for face_idx, face in enumerate(faces):
        for vertex in face:
            vertex_to_faces[vertex].append(face_idx)

    adjacency = set()  # Use a set to avoid duplicate pairs

    # Step 2: Create adjacency list from the vertex-to-face map
    for adjacent_faces in vertex_to_faces.values():
        # Sort face indices to avoid creating duplicate (a, b) and (b, a) pairs
        for i in range(len(adjacent_faces)):
            for j in range(i, len(adjacent_faces)):  # Ensure adj_face_idx > face_idx
                adjacency.add((adjacent_faces[i], adjacent_faces[j]))

    # Convert the set to a numpy array for consistency
    return np.array(list(adjacency))


def split_mesh_into_connected_components(
    verts, faces, return_face_mapping=False, return_vertex_mapping=False
):
    adjacency = vertex_adjacency(faces)
    components = connected_components(adjacency, min_len=0)
    print(len(components), "connected components found.")

    # # store connected components
    # if len(components) > 2:
    #     test_folder = "/scicore/home/engel0006/GROUP/pool-engel/Lorenz/MemBrain_stats/Phycobilisomes/test_meshes/"
    #     for i, component in enumerate(components):
    #         component_faces = faces[np.isin(np.arange(len(faces)), component)]
    #         component_mesh = trimesh.Trimesh(verts, component_faces)
    #         component_mesh.export(test_folder + f"component_{i}.obj")
    #     exit()

    component_face_idcs = [
        np.argwhere(np.isin(np.arange(len(faces)), component)).flatten()
        for component in components
    ]
    # if only one component, add unused faces to the last component
    if len(component_face_idcs) == 1:
        unused_faces = np.setdiff1d(np.arange(len(faces)), component_face_idcs[0])
        component_face_idcs[-1] = np.concatenate(
            [component_face_idcs[-1], unused_faces]
        )

    component_faces = [
        faces[component_face_idx] for component_face_idx in component_face_idcs
    ]

    forward_face_mapping = {
        face_idx: (component_idx, component_face_idx)
        for component_idx, cur_component_face_idcs in enumerate(component_face_idcs)
        for component_face_idx, face_idx in enumerate(cur_component_face_idcs)
    }
    reverse_face_mapping = {
        (component_idx, component_face_idx): face_idx
        for face_idx, (
            component_idx,
            component_face_idx,
        ) in forward_face_mapping.items()
    }

    resorted_meshes = [
        resort_mesh(verts, component_face, return_mapping=True)
        for component_face in component_faces
    ]
    component_verts = [resorted_mesh[0] for resorted_mesh in resorted_meshes]
    component_faces = [resorted_mesh[1] for resorted_mesh in resorted_meshes]
    vertex_mappings = [resorted_mesh[2] for resorted_mesh in resorted_meshes]

    forward_vertex_mapping = {
        vertex_idx: (component_idx, cur_vertex_mapping[vertex_idx])
        for component_idx, cur_vertex_mapping in enumerate(vertex_mappings)
        for vertex_idx in cur_vertex_mapping
    }

    # Reverse vertex mapping: (component_idx, component_vertex_idx) -> original index
    reverse_vertex_mapping = {
        (component_idx, component_vertex_idx): vertex_idx
        for vertex_idx, (
            component_idx,
            component_vertex_idx,
        ) in forward_vertex_mapping.items()
    }

    check_mappings(
        verts,
        faces,
        component_verts,
        component_faces,
        forward_vertex_mapping,
        reverse_vertex_mapping,
        forward_face_mapping,
        reverse_face_mapping,
    )
    out = (component_verts, component_faces)
    if return_face_mapping:
        out += (
            forward_face_mapping,
            reverse_face_mapping,
        )
    if return_vertex_mapping:
        out += (
            forward_vertex_mapping,
            reverse_vertex_mapping,
        )
    return out
