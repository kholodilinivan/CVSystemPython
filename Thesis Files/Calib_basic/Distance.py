import cv2
import numpy as np
import os
from scipy.io import savemat  # 需要安装 scipy: pip install scipy

# 相机内参矩阵 K 和畸变系数 D
K = np.array([[610.6938947728503, 0.0, 959.7314124388965], [0.0, 610.5677694496611, 959.7389316997021], [0.0, 0.0, 1.0]], dtype=np.float32)
D = np.array([0.001698329277591275,-0.002884210649590748, 0.001992752522615301,-0.0004896599361001555], dtype=np.float32)

# 棋盘格尺寸
CHECKERBOARD = (9,6)
per_grid_size = 116

# 准备对象点
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * per_grid_size
# 移动原点至棋盘格中心
center_x = (CHECKERBOARD[0] - 1) / 2.0 * per_grid_size
center_y = (CHECKERBOARD[1] - 1) / 2.0 * per_grid_size
objp[:, 0] -= center_x
objp[:, 1] -= center_y

image_paths = ['test_image1.jpg', 'test_image2.jpg']

# 保存参数的目录
output_dir = 'calibration_results'
os.makedirs(output_dir, exist_ok=True)

for fname in image_paths:
    img = cv2.imread(fname)
    if img is None:
        print(f"无法读取图像 {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 寻找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        # 对检测到的角点进行细化
        corners_refined = cv2.cornerSubPix(gray, corners, (40,40), (-1, -1),
                                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        # 使用 solvePnP 计算位姿
        _, rvec, tvec, _ = cv2.solvePnPRansac(objp, corners_refined, K, D)

        # 将旋转向量转换为旋转矩阵
        R, _ = cv2.Rodrigues(rvec)

        # 保存参数

        # 保存为文本文件
        txt_file_path = os.path.join(output_dir, f"{fname.split('.')[0]}_calib_params.txt")
        with open(txt_file_path, 'w') as f:
            f.write(f"图像名称: {fname}\n")
            f.write("相机内参矩阵 camera_matrix:\n")
            f.write(np.array2string(K, precision=6, separator=', '))
            f.write("\n\n畸变系数 dist_coeffs:\n")
            f.write(np.array2string(D, precision=6, separator=', '))
            f.write("\n\n旋转向量 rvecs:\n")
            f.write(np.array2string(rvec, precision=6, separator=', '))
            f.write("\n\n平移向量 tvecss:\n")
            f.write(np.array2string(tvec, precision=6, separator=', '))
            f.write("\n\n旋转矩阵 R:\n")
            f.write(np.array2string(R, precision=6, separator=', '))

        # 保存为 NumPy 的.npz 文件
        npz_file_path = os.path.join(output_dir, f"{fname.split('.')[0]}_calib_params.npz")
        np.savez(npz_file_path, camera_matrix=K, dist_coeffs=D, rvecs=rvec, tvecs=tvec, R=R)


        # 计算距离
        if fname == image_paths[0]:
            distance = -tvec[0][0]
        elif fname == image_paths[1]:
            distance = tvec[2][0]
        print(f"{fname} The distance of the origin of the checkerboard grid from the camera is：{distance:.4f} mm")
    else:
        print(f"{fname} 中未找到棋盘格。")