import cv2 # OpenCV库，核心图像处理功能
import numpy as np # 数值计算库
import glob # 文件路径匹配
import os # 操作系统功能

# ==================== 用户参数配置 ====================
CHECKERBOARD = (9,6)  # 棋盘格角点数（内部角点，例如10x7方格对应(9,6)）
size_per_grid = 116  # 棋盘格格子物理尺寸（单位：毫米）
calib_image_dir = 'D:/Git/CVSystemPython/Calibration/Calibration_images/*.jpg'  # 标定图像路径
output_dir = 'D:/Git/CVSystemPython/Calibration/Calibration_results'  # 结果保存目录


# ==================== 标定主代码 ====================
def main():
    # 检查 OpenCV 版本
    assert cv2.__version__[0] == '4', '需要 OpenCV >= 4.0'

    # 准备输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 生成三维角点坐标
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * size_per_grid

    # 检测角点
    objpoints, imgpoints = [], []
    _img_shape = None
    images = glob.glob(calib_image_dir)

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue

        # 检查图像尺寸一致性
        if _img_shape is None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "所有图像尺寸必须一致"

        # 角点检测
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
            gray, CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1),
                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
            objpoints.append(objp)
            imgpoints.append(corners)

    # 标定鱼眼相机
    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]

    rms, _, _, _, _ = cv2.fisheye.calibrate(
        objpoints, imgpoints, (1800,1800), K, D, rvecs, tvecs,
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )

    # 打印结果
    print(f"Found {N_OK} valid images for calibration")
    print(f"DIM={_img_shape[::-1]}")
    print(f"K=np.array({K.tolist()})")
    print(f"D=np.array({D.tolist()})")
    print(f"校准成功！RMS误差: {rms:.4f}")

    # ==================== 保存结果 ====================
    # 1. 保存为文本文件
    with open(os.path.join(output_dir, 'calibration_result.txt'), 'w') as f:
        f.write(f"标定时间: {np.datetime64('now')}\n")
        f.write(f"棋盘格参数: {CHECKERBOARD} (内部角点数), 格子尺寸: {size_per_grid} mm\n")
        f.write(f"图像尺寸 (WxH): {_img_shape[::-1]}\n")
        f.write(f"RMS误差: {rms:.4f} 像素\n\n")
        f.write("内参矩阵 camera_matrix:\n")  # 修改名称
        f.write(np.array2string(K, precision=6, separator=', '))
        f.write("\n\n畸变系数 dist_coeffs:\n")  # 修改名称
        f.write(np.array2string(D, precision=6, separator=', '))

    # 2. 保存为NumPy文件
    np.savez(os.path.join(output_dir, 'calibration_params.npz'),
             camera_matrix=K,  # 修改键名
             dist_coeffs=D,  # 修改键名
             image_size=_img_shape[::-1],
             rms=rms,
             checkerboard=CHECKERBOARD,
             grid_size=size_per_grid,)



if __name__ == '__main__':
    main()