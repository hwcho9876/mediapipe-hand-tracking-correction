"""Image-width coordinates: (x_norm, y_norm * height/width, z_norm)."""
import numpy as np

PALM = np.array([0, 5, 9, 13, 17])
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
         (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
         (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
EDGE_A = np.array([a for a, b in EDGES])
EDGE_B = np.array([b for a, b in EDGES])


def valid_points(points):
    return points is not None and np.shape(points) == (21, 3) and np.isfinite(points).all()


def center(points):
    return np.median(points[PALM], axis=0)


def palm_scale(points):
    # z is a relative model estimate, not camera distance in metres.
    return max(float(np.median(np.linalg.norm(points[[5, 9, 13, 17]] - points[0], axis=1))), 0.02)


def max_distance(a, b):
    return float(np.max(np.linalg.norm(a - b, axis=1)))


def to_width_coordinates(points, width, height):
    result = np.asarray(points, dtype=float).copy()
    result[:, 1] *= height / width
    return result


def to_normalized(points, width, height):
    result = np.asarray(points, dtype=float).copy()
    result[:, 1] *= width / height
    return result
