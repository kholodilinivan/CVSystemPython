import cv2
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize, disk
from sklearn.cluster import DBSCAN
from scipy.spatial import distance_matrix
import networkx as nx
import matplotlib.pyplot as plt

def las_segm(image):
    """
    Segment laser-like red lines and return a binary mask (0/1, uint8).
    Accepts either a file path (str) or a BGR numpy array as input.
    Returns:
        line_mask1: np.uint8 binary mask with detected line regions = 1
    """

    # ---------- helpers ----------
    def hsv_threshold(image, lower, upper, min_size=50, max_size=500, se_size=7):
        # Validate input image
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a numpy array")
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))

        # 形态学操作
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))
        mask = cv2.dilate(mask, se)
        mask = cv2.erode(mask, se)

        # 连通域筛选
        labeled = label(mask > 0)
        cleaned = np.zeros_like(mask, dtype=bool)
        for region in regionprops(labeled):
            if min_size <= region.area <= max_size:
                cleaned[labeled == region.label] = True
        return cleaned.astype(np.uint8)

    def _to_cv_hsv_bounds(lower_norm, upper_norm):
        """
        将 [0,1] 归一化 HSV 下界/上界转换为 OpenCV HSV (H:0-179, S:0-255, V:0-255).
        输入: lower_norm/upper_norm = [h, s, v] in [0,1]
        返回: lower_cv/upper_cv = [H, S, V] (uint8)
        """
        def clamp01(x):
            return max(0.0, min(1.0, float(x)))

        lh, ls, lv = (clamp01(lower_norm[0]), clamp01(lower_norm[1]), clamp01(lower_norm[2]))
        uh, us, uv = (clamp01(upper_norm[0]), clamp01(upper_norm[1]), clamp01(upper_norm[2]))
        lower_cv = [int(round(lh * 179)), int(round(ls * 255)), int(round(lv * 255))]
        upper_cv = [int(round(uh * 179)), int(round(us * 255)), int(round(uv * 255))]
        return lower_cv, upper_cv

    # ===================== 灰度阈值 =====================
    def gray_threshold(image, threshold=248, min_size=80, max_size=600, se_size=7):
        # Validate input image
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a numpy array")
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = (gray > threshold).astype(np.uint8)

        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))
        mask = cv2.dilate(mask, se)
        mask = cv2.erode(mask, se)

        # 连通域筛选
        labeled = label(mask > 0)
        cleaned = np.zeros_like(mask, dtype=bool)
        for region in regionprops(labeled):
            if min_size <= region.area <= max_size:
                cleaned[labeled == region.label] = True
        return cleaned.astype(np.uint8)

    # ===================== 线状物体检测和保留 =====================
    def filter_laser_lines(binary_img, min_length=30, max_width=10, min_aspect_ratio=3.0, solidity_threshold=0.8):
        """
        从二值图像中检测并保留激光线段

        参数:
            binary_img: 二值图像（包含激光线段和其他噪声）
            min_length: 激光线段最小长度
            max_width: 激光线段最大宽度
            min_aspect_ratio: 最小长宽比
            solidity_threshold: 实心度阈值（激光线段通常比较实心）

        返回:
            laser_mask: 只包含激光线段的二值图像
        """
        # 连通域分析
        labeled = label(binary_img > 0)
        laser_mask = np.zeros_like(binary_img, dtype=np.uint8)

        laser_regions = []

        for region in regionprops(labeled):
            # 获取区域的基本属性
            minr, minc, maxr, maxc = region.bbox
            height = maxr - minr
            width = maxc - minc

            # 计算主要特征
            length = max(height, width)  # 主要尺寸
            short_side = min(height, width)  # 次要尺寸

            # 避免除以零
            if short_side > 0:
                aspect_ratio = length / short_side
            else:
                aspect_ratio = 0

            area = region.area
            perimeter = region.perimeter
            solidity = region.solidity  # 实心度（区域面积/凸包面积）

            # 计算方向（主要特征向量）
            orientation = region.orientation

            # 激光线段的判断条件
            is_laser_line = (length >= min_length and
                             short_side <= max_width and
                             aspect_ratio >= min_aspect_ratio and
                             solidity >= solidity_threshold)

            if is_laser_line:
                # 保留这个激光线段
                laser_mask[labeled == region.label] = 1
                laser_regions.append({
                    'length': length,
                    'width': short_side,
                    'aspect_ratio': aspect_ratio,
                    'solidity': solidity,
                    'orientation': orientation,
                    'area': area
                })

                print(f"检测到激光线段: 长度={length:.1f}, 宽度={short_side:.1f}, "
                      f"长宽比={aspect_ratio:.2f}, 实心度={solidity:.3f}, "
                      f"方向角={np.degrees(orientation):.1f}°")

        return laser_mask, laser_regions

    # ===================== 骨架提取 + DBSCAN + 最小生成树 =====================
    def skeleton_and_centerline(binary_img, eps=20, min_samples=20):
        # 骨架
        skeleton = skeletonize(binary_img > 0)
        y, x = np.where(skeleton)
        points = np.vstack((x, y)).T

        # DBSCAN 聚类
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = db.labels_

        plt.imshow(binary_img, cmap='gray')
        plt.title("Laser Centerline")
        for lbl in np.unique(labels):
            if lbl == -1:  # 噪声
                continue
            cluster_points = points[labels == lbl]

            # 构建图，点间距小于30连边
            D = distance_matrix(cluster_points, cluster_points)
            G = nx.Graph()
            for i in range(len(cluster_points)):
                for j in range(i + 1, len(cluster_points)):
                    if D[i, j] < 30:
                        G.add_edge(i, j, weight=D[i, j])

            # 最小生成树
            T = nx.minimum_spanning_tree(G)
            for edge in T.edges():
                p1, p2 = cluster_points[edge[0]], cluster_points[edge[1]]
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', linewidth=2)
        plt.show()

    # ===================== MAIN PROCESSING LOGIC =====================
    
    # Handle both file path and numpy array input
    if isinstance(image, str):
        # If input is a file path, load the image
        image = cv2.imread(image)
        if image is None:
            raise FileNotFoundError(f"无法加载图片: {image}")
    
    # Validate the input image
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a numpy array or a valid image file path")
    
    if image.size == 0:
        raise ValueError("Input image is empty")
    
    print(f"Processing image: shape={image.shape}, dtype={image.dtype}")

    # ------- 方法1：强红（带较大饱和/亮度下界），min/max: 100/900 -------
    hue, saturation, value = 0.97, 0.95, 0.97
    hue_offset, sat_offset, val_offset = 0.05, 0.95, 0.05
    lower_red1 = [hue - hue_offset, max(saturation - sat_offset, 0), max(value - val_offset, 0)]
    upper_red1 = [1.0, 1.0, 1.0]
    lower_cv1, upper_cv1 = _to_cv_hsv_bounds(lower_red1, upper_red1)
    cleanedImage1 = hsv_threshold(image, lower_cv1, upper_cv1, min_size=30, max_size=300, se_size=7)

    # ------- 方法2：偏宽容的红色范围，min/max: 60/500 -------
    hue2, saturation2, value2 = 0.95, 0.95, 0.95
    hue_offset2, sat_offset2, val_offset2 = 0.05, 0.05, 0.95
    lower_red2 = [hue2 - hue_offset2, max(saturation2 - sat_offset2, 0),
                  max(value2 - val_offset2, 0)]
    upper_red2 = [1.0, 1.0, 1.0]
    lower_cv2, upper_cv2 = _to_cv_hsv_bounds(lower_red2, upper_red2)
    cleanedImage2 = hsv_threshold(image, lower_cv2, upper_cv2, min_size=100, max_size=300, se_size=7)

    # ------- 方法3：纯红（极窄 H 窗口，S 高，V 也较高），min/max: 100/500 -------
    hue3, saturation3, value3 = 0.0, 1.0, 0.6
    hue_offset3, sat_offset3, val_offset3 = 0.001, 0.001, 0.55
    lower_red3 = [0.0, max(saturation3 - sat_offset3, 0), max(value3 - val_offset3, 0)]
    upper_red3 = [hue3 + hue_offset3, 1.0, 1.0]
    lower_cv3, upper_cv3 = _to_cv_hsv_bounds(lower_red3, upper_red3)
    cleanedImage3 = hsv_threshold(image, lower_cv3, upper_cv3, min_size=60, max_size=300, se_size=7)

    # ------- 方法4：偏红的泛光（较低 S/V 下界），min/max: 100/400 -------
    hue4, saturation4, value4 = 0.8, 0.3, 0.3
    lower_red4 = [hue4, max(saturation4, 0), max(value4, 0)]
    upper_red4 = [1.0, 1.0, 1.0]
    lower_cv4, upper_cv4 = _to_cv_hsv_bounds(lower_red4, upper_red4)
    cleanedImage4 = hsv_threshold(image, lower_cv4, upper_cv4, min_size=60, max_size=600, se_size=7)

    # ------- 方法5：高亮反射（偏白黄到浅红），min/max: 100/400 -------
    lower_red5 = [0.1, 0.1, 0.7]
    upper_red5 = [0.3, 0.3, 0.9]
    lower_cv5, upper_cv5 = _to_cv_hsv_bounds(lower_red5, upper_red5)
    cleanedImage5 = hsv_threshold(image, lower_cv5, upper_cv5, min_size=60, max_size=500, se_size=7)

    cleaned2 = gray_threshold(image, threshold=248)

    # 合并结果
    combined = (cleanedImage1 | cleanedImage2 | cleanedImage3 | cleanedImage4 | cleanedImage5).astype(np.uint8)

    line_mask1, laser_regions = filter_laser_lines(combined,
                                    min_length=15,
                                    max_width=50,
                                    min_aspect_ratio=4,
                                    solidity_threshold=0.3)

    return line_mask1


# This block only runs when the script is executed directly, not when imported
if __name__ == "__main__":
    # Test code when running this file directly
    try:
        image = cv2.imread('img/2_1.jpg')  # load test image
        if image is None:
            raise FileNotFoundError("无法加载测试图片 'img/2_1.jpg'")
        
        result = las_segm(image)
        print(f"Result shape: {result.shape}, dtype: {result.dtype}")
        
        # Display the result
        plt.imshow(result, cmap='gray')
        plt.title("Laser Segmentation Result")
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")