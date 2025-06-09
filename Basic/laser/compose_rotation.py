
import numpy as np

def compose_rotation(x, y, z):
    x = np.deg2rad(x)
    y = np.deg2rad(y)
    z = np.deg2rad(z)

    X = np.array([[1, 0, 0],
                  [0, np.cos(x), -np.sin(x)],
                  [0, np.sin(x),  np.cos(x)]])
    Y = np.array([[ np.cos(y), 0, np.sin(y)],
                  [0, 1, 0],
                  [-np.sin(y), 0, np.cos(y)]])
    Z = np.array([[np.cos(z), -np.sin(z), 0],
                  [np.sin(z),  np.cos(z), 0],
                  [0, 0, 1]])
    return Z @ Y @ X
