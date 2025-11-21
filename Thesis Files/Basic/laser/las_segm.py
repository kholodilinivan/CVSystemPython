
import cv2
import numpy as np

def las_segm(image_path):
    """
    加载图像并提取红色激光条纹区域，返回二值图像掩码（红色区域为1）。
    """
    image = cv2.imread(image_path)
    if image is None:
        print("[ERROR] Cannot load image:", image_path)
        return np.zeros((480, 640), dtype=np.uint8)

    # 转换为 HSV 颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 红色激光分成两个范围（低H + 高H）
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # 创建掩码
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # 可选：形态学操作去噪
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 转为0-1二值图
    binary_mask = (mask > 0).astype(np.uint8)

    return binary_mask
