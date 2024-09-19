import numpy as np
from scipy.spatial.distance import cdist
from membrain_stats.utils.mesh_utils import (
    find_closest_vertices,
    split_mesh_into_connected_components,
)


class GeodesicDistanceSolver:
    def __init__(self, verts, faces, method="fast"):
        self.verts = verts
        self.faces = faces
        self.method = method

        self.components = split_mesh_into_connected_components(
            verts, faces, return_face_mapping=True, return_vertex_mapping=True
        )
        self.forward_vertex_mapping = self.components[4]
        self.reverse_vertex_mapping = self.components[5]
        self.solvers = []
        if self.method == "exact":
            from pygeodesic import geodesic

            for component_verts, component_faces in zip(*self.components[:2]):
                geoalg = geodesic.PyGeodesicAlgorithmExact(
                    component_verts, component_faces
                )
                self.solvers.append(geoalg)
        elif self.method == "fast":
            import potpourri3d as pp3d

            for component_verts, component_faces in zip(*self.components[:2]):
                solver = pp3d.MeshHeatMethodDistanceSolver(
                    V=component_verts, F=component_faces
                )
                self.solvers.append(solver)

    def compute_geod_distance_matrix(self, point_idx):
        # map point index to component index
        component_idx, component_point_idx = self.forward_vertex_mapping[point_idx]
        solver = self.solvers[component_idx]
        if self.method == "exact":
            distances, _ = solver.geodesicDistances(
                np.array([component_point_idx]), np.arange(len(self.verts))
            )
        elif self.method == "fast":
            distances = solver.compute_distance(component_point_idx)
        distances[component_point_idx] = -1

        # map back to original vertex indices
        full_distances = np.full(len(self.verts), np.inf)
        reverse_idcs = np.array(
            [
                self.reverse_vertex_mapping[(component_idx, idx)]
                for idx in np.arange(len(distances))
            ]
        )
        full_distances[reverse_idcs] = distances
        return full_distances


def compute_geodesic_distance_matrix(
    verts: np.ndarray,
    faces: np.ndarray,
    point_coordinates: np.ndarray,
    point_coordinates_target: np.ndarray = None,
    method: str = "exact",
    return_mesh_distances: bool = False,
    infinite_distance_for_pointless_areas: bool = True,
):
    """Compute the geodesic distance matrix between two sets of points on a mesh.

    Parameters
    ----------
    verts : np.ndarray
        The vertices of the mesh.
    faces : np.ndarray
        The faces of the mesh.
    point_coordinates1 : np.ndarray
        The coordinates of the first set of points for which the geodesic distances should be computed.
    point_coordinates2 : np.ndarray
        The coordinates of the second set of points for which the geodesic distances should be computed.
    method : str
        The method to use for computing the geodesic distances. Can be either "exact" or "fast".

    Returns
    -------
    np.ndarray
        The geodesic distance matrix between the points.

    Note
    -------
    This function has been reproduced from
    github.com/cellcanvas/surforama/blob/main/src/surforama/utils/stats.py

    """
    solver = GeodesicDistanceSolver(verts, faces, method=method)
    if point_coordinates_target is None:
        point_coordinates_target = point_coordinates
    point_idcs = find_closest_vertices(verts, point_coordinates).tolist()
    point_idcs_target = find_closest_vertices(verts, point_coordinates_target).tolist()
    distance_matrix = np.zeros((len(point_idcs), len(point_idcs_target))).astype(
        np.float32
    )
    if return_mesh_distances:
        mesh_distances = []
    for i, point_idx in enumerate(point_idcs):
        distances = solver.compute_geod_distance_matrix(point_idx)
        cur_mesh_distances = distances.copy()
        distances = distances[point_idcs_target]
        if return_mesh_distances:
            if infinite_distance_for_pointless_areas and np.all(np.isinf(distances)):
                cur_mesh_distances = np.full(len(verts), np.inf)
            mesh_distances.append(cur_mesh_distances)
        distance_matrix[i] = distances
    if return_mesh_distances:
        mesh_distances = np.array(mesh_distances)
        return distance_matrix, mesh_distances
    return distance_matrix


def compute_euclidean_distance_matrix(
    verts: np.ndarray,
    point_coordinates: np.ndarray,
    point_coordinates_target: np.ndarray = None,
    return_mesh_distances: bool = False,
):
    """Compute the euclidean distance matrix between two sets of points.

    Parameters
    ----------
    verts : np.ndarray
        The vertices of the mesh.
    point_coordinates : np.ndarray
        The coordinates of the points for which the euclidean distances should be computed.
    point_coordinates_target : np.ndarray
        The coordinates of the target points for which the euclidean distances should be computed.

    Returns
    -------
    np.ndarray
        The euclidean distance matrix between the points.

    """
    if point_coordinates_target is None:
        point_coordinates_target = point_coordinates
    distance_matrix = cdist(point_coordinates, point_coordinates_target)

    if return_mesh_distances:
        mesh_distances = np.zeros((len(verts), len(point_coordinates)))
        for i, point in enumerate(point_coordinates):
            mesh_distances[:, i] = np.linalg.norm(verts - point, axis=1)
        mesh_distances = mesh_distances.T
        return distance_matrix, mesh_distances
    return distance_matrix
