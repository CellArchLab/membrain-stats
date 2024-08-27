import numpy as np
from membrain_stats.utils.mesh_utils import find_closest_vertices


class GeodesicDistanceSolver:
    def __init__(self, verts, faces, method="fast"):
        self.verts = verts
        self.faces = faces
        self.method = method

        if method == "exact":
            import pygeodesic as geodesic
            self.geoalg = geodesic.PyGeodesicAlgorithmExact(verts, faces)
        elif method == "fast":
            import potpourri3d as pp3d
            self.solver = pp3d.MeshHeatMethodDistanceSolver(V=verts, F=faces)
    
    def compute_geod_distance_matrix(self, point_idx):
        if self.method == "exact":
            distances, _ = self.geoalg.geodesicDistances(
                np.array([point_idx]), np.arange(len(self.verts))
            )
        elif self.method == "fast":
            distances = self.solver.compute_distance(point_idx)

        distances[point_idx] = 1e5
        return distances


def compute_geodesic_distance_matrix(
        verts: np.ndarray,
        faces: np.ndarray,
        point_coordinates: np.ndarray,
        point_coordinates_target: np.ndarray = None,
        method: str = "exact",
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
    point_idcs = [
        find_closest_vertices(verts, point) for point in point_coordinates
    ]
    point_idcs_target = [
        find_closest_vertices(verts, point) for point in point_coordinates_target
    ]

    distance_matrix = np.zeros((len(point_idcs), len(point_idcs_target))).astype(
        np.float32
    )
    for i, point_idx in enumerate(point_idcs):
        distances = solver.compute_geod_distance_matrix(point_idx)
        distances = distances[point_idcs_target]
        distance_matrix[i] = distances[:, 0]

    return distance_matrix