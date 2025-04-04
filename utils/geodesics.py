import trimesh
import numpy as np
import trimesh.visual


def get_all_face_vertices(verts, faces):
    return verts[faces]


def get_all_local_face_coordinates(verts, faces):
    """
    Calculate the local coordinates of each face in a given mesh.

    Args:
        verts (numpy.ndarray): Array of vertex coordinates.
        faces (numpy.ndarray): Array of face indices.

    Returns:
        numpy.ndarray: Array of local face coordinates.
    """
    local_faces = []
    for face in faces:
        a, b, c = face
        ab = ((verts[b, :] - verts[a, :]) ** 2).sum().sqrt()
        bc = ((verts[c, :] - verts[b, :]) ** 2).sum().sqrt()
        ac = ((verts[a, :] - verts[c, :]) ** 2).sum().sqrt()

        s = (ab + bc + ac) / 2  # semi-perimeter
        area = (s * (s - ab) * (s - bc) * (s - ac)) ** 0.5
        h = 2 * area / ab  # height of the triangle
        w = max(0, (ac**2 - h**2) ** 0.5)  # width of the triangle

        # local_a is always (0, 0)
        local_b = np.array([ab, 0])
        local_c = np.array([w, h])
        local_faces.append(np.array([local_b, local_c]))
    return np.array(local_faces)


def get_all_face_edges(faces):
    """
    Calculate the edges of each face in a given mesh.

    Args:
        faces (numpy.ndarray): Array of face indices.

    Returns:
        numpy.ndarray: Array of face edges.
    """
    face_edges = []
    for face in faces:
        a, b, c = face
        face_edges.append(np.array([[a, b], [b, c], [c, a]]))
    return np.array(face_edges)


def cartesian_to_barycentric_coordinates(point, triangle):
    """
    Calculate the barycentric coordinates of a point in a triangle.

    Parameters
    ----------
    point : np.array
        The point to calculate the barycentric coordinates for.
    triangle : np.array
        Position of the vertices of the triangle to calculate the barycentric
        coordinates in.

    Returns
    -------
    np.array
        The barycentric coordinates of the point in the triangle.
    """
    v0, v1, v2 = triangle
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = point - v0

    d00 = np.dot(v0v1, v0v1)
    d01 = np.dot(v0v1, v0v2)
    d11 = np.dot(v0v2, v0v2)
    d20 = np.dot(v0p, v0v1)
    d21 = np.dot(v0p, v0v2)

    denom = d00 * d11 - d01 * d01

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w

    return np.array([u, v, w])


def barycentryc_to_cartesian_coordinates(barycentric, triangle):
    """
    Calculate the cartesian coordinates of a point in a triangle.

    Parameters
    ----------
    barycentric : np.array
        The barycentric coordinates of the point in the triangle.
    triangle : np.array
        Position of the vertices of the triangle to calculate the cartesian
        coordinates in.

    Returns
    -------
    np.array
        The cartesian coordinates of the point.
    """
    v0, v1, v2 = triangle
    return barycentric[0] * v0 + barycentric[1] * v1 + barycentric[2] * v2


def face_to_barycentric_coordinates(point_in_face_coordinates, triangle):
    """
    Calculate the barycentric coordinates of a point in a triangle, whose
    coordinates are given in face coordinates. Face coordinates are essentially
    the coordinates of the point in the triangle's local coordinate system,
    which is centered at the triangle's first vertex.

    Args:
        point_in_face_coordinates: np.array
            2D coordinates of a point in the triangle's local coordinate system.
        triangle: np.array
            3D position of the vertices of the triangle to calculate the
            barycentric coordinates in.
    """
    w = point_in_face_coordinates[1] / triangle[2, 1]
    w = np.clip(w, 0.0, 1.0)

    v = (point_in_face_coordinates[0] - w * triangle[2, 0]) / triangle[1, 0]
    v = np.clip(v, 0.0, 1.0 - w)

    u = 1.0 - v - w
    u = np.clip(u, 0.0, 1.0)
    return np.array([u, v, w])


def is_in_trinagle(barycentric_point):
    # Original implementation wasn't checking if <=1, not sure why as that
    # is a requirement for barycentric coordinates
    return np.all(barycentric_point >= 0) and np.all(barycentric_point <= 1)


def trace_in_face_barycentric(
    vertices,
    faces,
    all_face_edges,
    all_face_coordinates,
    face_id,
    start_point_barycentric,
    vector_barycentric,
    vector_cartesian_length,
    vector_cartesian_direction,
    hittable_edges_in_face,
    trace_eps_loose=1e-9,
):
    if sum(start_point_barycentric) < 0.5:
        raise ValueError("Bad start barycentric point")

    end_point_barycentric = start_point_barycentric + vector_barycentric

    if is_in_trinagle(end_point_barycentric):  # found final point
        return {
            "terminated": True,
            "face_id": face_id,
            "end_point_barycentric": end_point_barycentric,
            "end_point_cartesian": barycentryc_to_cartesian_coordinates(
                end_point_barycentric, all_face_coordinates[face_id]
            ),
            "incoming_direction_to_point_cartesian": vector_cartesian_direction,
            # "trace_vector_in_halfedge_length": 0
        }
    else:
        # The vector did not end in this triangle.
        # Pick an appropriate point along some edge
        t_ray = np.infty
        cross_edge = None
        for i in range(3):
            if not hittable_edges_in_face[i] or vector_barycentric[i] >= 0:
                # Not sure the second condition is correct
                continue

            t_ray_this_raw = -start_point_barycentric[i] / vector_barycentric[i]

            if t_ray_this_raw < t_ray:
                # This is the new closest intersection
                t_ray = t_ray_this_raw
                cross_edge = all_face_edges[face_id, i]
                # TODO: finish implementation


if __name__ == "__main__":
    # Load the mesh
    # mesh = trimesh.load_mesh("/data2/simple_meshes/spot.obj", process=False)

    mesh = trimesh.primitives.Box([2, 2, 2])

    mesh.visual = trimesh.visual.color.ColorVisuals(
        mesh, vertex_colors=np.array([0, 0, 200])
    )

    # Generate random segments
    segments = np.random.random((100, 2, 3))
    path = trimesh.load_path(
        segments, colors=np.stack([[255, 0, 0]] * segments.shape[0])
    )

    # Sample a point on the mesh
    # point, _ = trimesh.sample.sample_surface(mesh, 1)
    point = np.array([1, 0.5, 0.5])
    point_sphere = trimesh.primitives.Sphere(radius=0.05, center=point)
    point_sphere.visual = trimesh.visual.color.ColorVisuals(
        point_sphere, vertex_colors=np.array([255, 0, 0])
    )

    face_verts = get_all_face_vertices(mesh.vertices, mesh.faces)

    point_bary = cartesian_to_barycentric_coordinates(point, face_verts[-1, :, :])

    point_cart = barycentryc_to_cartesian_coordinates(point_bary, face_verts[-1, :, :])

    print(point_cart)
    trimesh.Scene([mesh, path, point_sphere]).show()
