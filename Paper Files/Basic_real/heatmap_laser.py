
import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_laser_heatmap(mask_path="debug_mask.png"):
    img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("[ERROR] Cannot load image:", mask_path)
        return

    # 构建激光热度图
    heatmap = np.zeros_like(img, dtype=np.uint16)
    rows, cols = np.where(img > 10)
    for r, c in zip(rows, cols):
        heatmap[r, c] += 1

    # 可视化热力图
    plt.figure(figsize=(12, 6))
    plt.imshow(heatmap, cmap="hot")
    plt.title("Laser Pixel Heatmap")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.colorbar(label="Laser Pixel Count")
    plt.savefig("laser_heatmap.png")
    print("[INFO] Heatmap saved as laser_heatmap.png")

if __name__ == "__main__":
    visualize_laser_heatmap()
