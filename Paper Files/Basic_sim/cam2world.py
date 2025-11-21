
import numpy as np

def cam2world(m, ocam_model):
    ss = ocam_model["ss"]
    xc, yc = ocam_model["xc"], ocam_model["yc"]
    c, d, e = ocam_model["c"], ocam_model["d"], ocam_model["e"]

    # 像素归一化
    A = np.array([[c, d], [e, 1]])
    T = np.array([[xc], [yc]]).reshape(2, 1)
    m = np.dot(np.linalg.inv(A), (m.reshape(2, 1) - T))

    # 半径距离
    r = np.sqrt(m[0]**2 + m[1]**2)

    # 用 ss 多项式计算 z 分量
    z = np.polyval(ss[::-1], r)
    M = np.vstack((m[0], m[1], z))
    M = M / np.linalg.norm(M)
    return M
