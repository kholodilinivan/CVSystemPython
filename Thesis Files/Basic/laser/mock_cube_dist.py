
import numpy as np

def mock_cube_dist(image, row_range, col_range, name="region"):
    rows = []
    height, width = image.shape

    row_start, row_end = max(0, row_range[0]), min(height, row_range[1])
    col_start, col_end = max(0, col_range[0]), min(width, col_range[1])

    for i in range(row_start, row_end):
        for j in range(col_start, col_end):
            if image[i, j] > 0:
                rows.append(i)

    if len(rows) == 0:
        print(f"[DEBUG] {name} avg_row: N/A (no pixels found)")
        return 0.0

    avg_row = np.mean(rows)
    print(f"[DEBUG] {name} avg_row: {avg_row:.2f}")

    # 最终拟合函数：distance = 3427.91 - 6.27 * row + 0.00156 * row^2
    distance = 3427.905 - 6.26994 * avg_row + 0.0015614 * avg_row ** 2
    return distance
