
import cv2
import numpy as np

def analyze_mask(path="debug_mask.png"):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("[ERROR] Cannot load image:", path)
        return

    rows, cols = np.where(img > 10)  # 阈值略大于0，避免图像噪点
    if len(rows) == 0:
        print("[DEBUG] No laser pixels found in mask.")
        return

    min_row, max_row = np.min(rows), np.max(rows)
    print(f"[DEBUG] Laser pixel row range: {min_row} to {max_row}")

    # 每行激光像素统计
    unique, counts = np.unique(rows, return_counts=True)
    print("[DEBUG] Laser pixel count by row (sample):")
    for r, c in zip(unique[:10], counts[:10]):
        print(f"  Row {r}: {c} pixels")

if __name__ == "__main__":
    analyze_mask()
