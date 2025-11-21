
import numpy as np

def print_laser_bounds(img_bin):
    """
    打印激光条纹在图像中出现的最小/最大行号，以及每行像素数分布。
    """
    rows, cols = np.where(img_bin > 0)
    if len(rows) == 0:
        print("[DEBUG] No laser pixels detected in image.")
        return

    min_row = np.min(rows)
    max_row = np.max(rows)
    print(f"[DEBUG] Laser pixel row range: {min_row} to {max_row}")

    # 行像素数统计
    unique, counts = np.unique(rows, return_counts=True)
    print("[DEBUG] Laser pixel count by row (sample):")
    for r, c in zip(unique[:10], counts[:10]):
        print(f"  Row {r}: {c} pixels")
