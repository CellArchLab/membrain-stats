import numpy as np

def resort_mesh(
        verts: np.ndarray, 
        faces: np.ndarray,):
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
    return new_verts, new_faces


def find_closest_vertices(verts, points):
    """Find the index of the closest vertex to a given point."""
    if len(points.shape) == 1:
        points = points[np.newaxis, :]
    distances = np.linalg.norm(verts[:, np.newaxis] - points, axis=2)
    return np.argmin(distances, axis=0)

