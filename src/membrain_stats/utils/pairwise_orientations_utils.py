import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R
from membrain_stats.utils.mesh_utils import find_closest_vertices


def project_to_mean_plane(inplane1, inplane2, mesh, pos1, pos2):
    average_pos = (pos1 + pos2) / 2
    closest_vertex = find_closest_vertices(mesh.vertices, np.array([average_pos]))[0]
    normal = mesh.vertex_normals[closest_vertex]
    inplane1 = inplane1 - np.dot(inplane1, normal) * normal
    inplane2 = inplane2 - np.dot(inplane2, normal) * normal
    return inplane1, inplane2


def normalize_vectors(vectors):
    if len(vectors.shape) == 1:
        return vectors / np.linalg.norm(vectors)
    return vectors / np.linalg.norm(vectors, axis=1)[:, None]


def pairwise_angle(
    inplane1: np.ndarray,
    inplane2: np.ndarray,
    c2_sym: bool = False,
    project_to_plane: bool = False,
    pos1: np.ndarray = None,
    pos2: np.ndarray = None,
    mesh: trimesh.Trimesh = None,
):
    if project_to_plane:
        if mesh is None or pos1 is None or pos2 is None:
            raise ValueError("Need to provide mesh and positions for projection.")
        inplane1, inplane2 = project_to_mean_plane(inplane1, inplane2, mesh, pos1, pos2)
    inplane1 = normalize_vectors(inplane1)
    inplane2 = normalize_vectors(inplane2)
    angle = np.arccos(np.dot(inplane1, inplane2))
    if c2_sym:
        angle = np.minimum(angle, np.pi - angle)
    return np.degrees(angle)


def angle_matrix(
    angles1: np.ndarray,
    angles2: np.ndarray,
    c2_symmetry: bool = False,
    pos1: np.ndarray = None,
    pos2: np.ndarray = None,
    mesh: np.ndarray = None,
    project_to_plane: bool = False,
):
    rotation1 = R.from_euler(seq="ZYZ", angles=angles1, degrees=True).inv()
    rotation2 = R.from_euler(seq="ZYZ", angles=angles2, degrees=True).inv()

    initial_normals = np.array([0, 0, 1])
    initial_inplane = np.array([0, 1, 0])

    normals1 = rotation1.apply(initial_normals.copy())
    inplane1 = rotation1.apply(initial_inplane.copy())

    normals2 = rotation2.apply(initial_normals.copy())
    inplane2 = rotation2.apply(initial_inplane.copy())

    normals1 = normalize_vectors(normals1)
    inplane1 = normalize_vectors(inplane1)

    normals2 = normalize_vectors(normals2)
    inplane2 = normalize_vectors(inplane2)

    angle_matrix = np.zeros((len(angles1), len(angles2)))
    for i in range(len(angles1)):
        for j in range(len(angles2)):
            if i == j:
                angle_matrix[i, j] = 0.0
            else:
                angle_matrix[i, j] = pairwise_angle(
                    inplane1[i],
                    inplane2[j],
                    c2_sym=c2_symmetry,
                    project_to_plane=project_to_plane,
                    mesh=mesh,
                    pos1=pos1[i],
                    pos2=pos2[j],
                )

    return angle_matrix
