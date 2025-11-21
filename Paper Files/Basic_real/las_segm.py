import cv2
import numpy as np
from skimage.measure import label, regionprops

def las_segm(image_or_path):
    """
    Segment laser-like red lines and return a binary mask (0/1, uint8).
    Accepts either a file path (str) or a BGR numpy array as input.
    Returns:
        line_mask1: np.uint8 binary mask with detected line regions = 1
    """

    # ---------- helpers ----------
    def hsv_threshold(image, lower, upper, min_size=50, max_size=500, se_size=7):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))

        # morphology
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))
        mask = cv2.dilate(mask, se)
        mask = cv2.erode(mask, se)

        # CC filter
        labeled = label(mask > 0)
        cleaned = np.zeros_like(mask, dtype=bool)
        for region in regionprops(labeled):
            if min_size <= region.area <= max_size:
                cleaned[labeled == region.label] = True
        return cleaned.astype(np.uint8)

    def _to_cv_hsv_bounds(lower_norm, upper_norm):
        # convert normalized [0..1] HSV to OpenCV HSV [H:0..179, S/V:0..255]
        def clamp01(x): return max(0.0, min(1.0, float(x)))
        lh, ls, lv = (clamp01(lower_norm[0]), clamp01(lower_norm[1]), clamp01(lower_norm[2]))
        uh, us, uv = (clamp01(upper_norm[0]), clamp01(upper_norm[1]), clamp01(upper_norm[2]))
        lower_cv = [int(round(lh * 179)), int(round(ls * 255)), int(round(lv * 255))]
        upper_cv = [int(round(uh * 179)), int(round(us * 255)), int(round(uv * 255))]
        return lower_cv, upper_cv

    def gray_threshold(image, threshold=248, min_size=80, max_size=600, se_size=7):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = (gray > threshold).astype(np.uint8)
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))
        mask = cv2.dilate(mask, se)
        mask = cv2.erode(mask, se)
        labeled = label(mask > 0)
        cleaned = np.zeros_like(mask, dtype=bool)
        for region in regionprops(labeled):
            if min_size <= region.area <= max_size:
                cleaned[labeled == region.label] = True
        return cleaned.astype(np.uint8)

    def filter_laser_lines(binary_img, min_length=30, max_width=10, min_aspect_ratio=3.0, solidity_threshold=0.8):
        labeled = label(binary_img > 0)
        laser_mask = np.zeros_like(binary_img, dtype=np.uint8)
        laser_regions = []
        for region in regionprops(labeled):
            minr, minc, maxr, maxc = region.bbox
            h = maxr - minr
            w = maxc - minc
            length = max(h, w)
            short_side = min(h, w)
            aspect_ratio = (length / short_side) if short_side > 0 else 0.0
            solidity = region.solidity

            is_laser_line = (length >= min_length and
                             short_side <= max_width and
                             aspect_ratio >= min_aspect_ratio and
                             solidity >= solidity_threshold)
            if is_laser_line:
                laser_mask[labeled == region.label] = 1
                laser_regions.append({
                    'length': float(length),
                    'width': float(short_side),
                    'aspect_ratio': float(aspect_ratio),
                    'solidity': float(solidity)
                })
        return laser_mask, laser_regions

    # ---------- load input ----------
    if isinstance(image_or_path, str):
        image = cv2.imread(image_or_path)
        if image is None:
            raise FileNotFoundError(f"无法加载图片: {image_or_path}")
    elif isinstance(image_or_path, np.ndarray):
        image = image_or_path
    else:
        raise TypeError("image_or_path 必须是图像路径字符串或 BGR 的 numpy.ndarray")

    # ---------- HSV-based masks ----------
    # Method 1
    lower_cv1, upper_cv1 = _to_cv_hsv_bounds(
        [0.97 - 0.05, max(0.95 - 0.95, 0), max(0.97 - 0.05, 0)],
        [1.0, 1.0, 1.0]
    )
    cleanedImage1 = hsv_threshold(image, lower_cv1, upper_cv1, min_size=30, max_size=300, se_size=7)

    # Method 2
    lower_cv2, upper_cv2 = _to_cv_hsv_bounds(
        [0.95 - 0.05, max(0.95 - 0.05, 0), max(0.95 - 0.95, 0)],
        [1.0, 1.0, 1.0]
    )
    cleanedImage2 = hsv_threshold(image, lower_cv2, upper_cv2, min_size=100, max_size=300, se_size=7)

    # Method 3
    lower_cv3, upper_cv3 = _to_cv_hsv_bounds(
        [0.0, max(1.0 - 0.001, 0), max(0.6 - 0.55, 0)],
        [0.0 + 0.001, 1.0, 1.0]
    )
    cleanedImage3 = hsv_threshold(image, lower_cv3, upper_cv3, min_size=100, max_size=300, se_size=7)

    # Method 4
    lower_cv4, upper_cv4 = _to_cv_hsv_bounds([0.92, 0.4, 0.4], [1.0, 1.0, 1.0])
    cleanedImage4 = hsv_threshold(image, lower_cv4, upper_cv4, min_size=100, max_size=400, se_size=7)

    # Method 5
    lower_cv5, upper_cv5 = _to_cv_hsv_bounds([0.1, 0.1, 0.7], [0.3, 0.3, 0.9])
    cleanedImage5 = hsv_threshold(image, lower_cv5, upper_cv5, min_size=60, max_size=500, se_size=7)

    # Optional (not used later, but kept for parity with your code)
    _ = gray_threshold(image, threshold=248)

    # ---------- combine masks & filter line-like regions ----------
    combined = (cleanedImage1 | cleanedImage2 | cleanedImage3 | cleanedImage4 | cleanedImage5).astype(np.uint8)
    line_mask1, _ = filter_laser_lines(
        combined,
        min_length=15,
        max_width=50,
        min_aspect_ratio=4.0,
        solidity_threshold=0.3
    )

    return line_mask1
