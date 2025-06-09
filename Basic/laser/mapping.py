
import numpy as np
from cam2world import cam2world
from compose_rotation import compose_rotation

def mapping(image, x_angle, y_angle, las_dist, ocam_model):
    height, width = image.shape
    Z = las_dist
    x1, y1 = [], []
    t = np.array([[0], [0], [Z]])
    r = compose_rotation(-x_angle, -y_angle, 0)
    r = np.hstack((r[:, :2], t))

    for i in range(height):
        if i % 50 == 0:
            print(f"[mapping] Processing row {i}/{height}")
        for j in range(width):
            if image[i, j] > 0:
                m = np.array([j, i])
                M = cam2world(m, ocam_model)  # unit direction
                P = Z * M                     # real 3D point
                a1 = P[0]*r[1,0] - P[1]*r[0,0]
                b1 = P[0]*r[1,1] - P[1]*r[0,1]
                c1 = P[0]*r[1,2] - P[1]*r[0,2]
                a2 = P[2]*r[0,0] - P[0]*r[2,0]
                b2 = P[2]*r[0,1] - P[0]*r[2,1]
                c2 = P[2]*r[0,2] - P[0]*r[2,2]
                denom = a1 * b2 - a2 * b1
                if denom == 0 or a1 == 0:
                    continue
                Y = (a2 * c1 - a1 * c2) / denom
                X = (-c1 - b1 * Y) / a1
                x1.append(-X)
                y1.append(Y)
    return np.array(x1), np.array(y1)
