
import cv2
import numpy as np
import matplotlib.pyplot as plt

from ocam_model_sim_matlab import get_ocam_model # camera model - matlab data
# from ocam_model_sim_python import get_ocam_model # camera model - python data

from las_segm import las_segm # Laser stripe segmentation - for sim function
from mapping import mapping # 3D coordinate mapping
from mock_cube_dist import mock_cube_dist  # Cube depth simulation
from laser_debug import print_laser_bounds # Laser boundary debugging

# === Configuration parameters ===
image_path = "image.jpg" # Enter the image path
x_angle = 1.5 # Laser plane X direction angle (degrees)
y_angle = -2.5 # Laser plane Y direction angle (degrees)
las_dist = 950 # Distance from laser to reference plane (mm)

CVsyst_x=0 # Camera coordinate system X origin
CVsyst_y=0 # Camera coordinate system Y origin

#=== Load images and models ===
img_bin = las_segm(image_path)
print_laser_bounds(img_bin)

ocam_model = get_ocam_model()

#=== Calculate the overall mapping point (visualization) ===
x1, y1 = mapping(img_bin, x_angle, y_angle, las_dist, ocam_model)

#=== Simulation depth estimation ===
'''
for regions selection Run file: get_pixel_data.py and click mouse to select Laser regions
next write it to the code below as: [column pixels], [row pixels]
if laser is located verticaly: vertical = 1, if laser is horisontal: vertical = 0
'''
C_left = mock_cube_dist(img_bin, x_angle, y_angle, las_dist, ocam_model, [634, 867], [504, 550], vertical = 1, name="Left Cube")
C_Up = mock_cube_dist(img_bin, x_angle, y_angle, las_dist, ocam_model, [326, 353], [687, 843], vertical = 0, name="Up Cube")
C_Right = mock_cube_dist(img_bin, x_angle, y_angle, las_dist, ocam_model, [488, 613], [1171, 1260], vertical = 1, name="Right Cube")

#=== Print the estimated result ===
print(f"Left Cube Distance Estimate: {C_left:.2f} mm")
print(f"Up Cube Distance Estimate: {C_Up:.2f} mm")
print(f"Right Cube Distance Estimate: {C_Right:.2f} mm")

#=== Visual image (saved and not displayed)===
plt.figure(figsize=(8, 6))
# plt.gca().set_xlim([-200, 200]) # set limits if image is noisy (show only working region)
# plt.gca().set_ylim([-200, 200]) # set limits if image is noisy (show only working region)

plt.scatter(x1, y1, s=3, c='b', label='Laser Intersections') # use for matlab calib
# plt.scatter(-x1, -y1, s=3, c='b', label='Laser Intersections') # use for python calib

plt.scatter([CVsyst_x], [-CVsyst_y], c='r', marker='*', s=100, label='Camera')
plt.title("3D Mapping of Laser Line Intersections")
plt.xlabel("X (mm)")
plt.ylabel("Y (mm)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("mapping_result.png")
plt.close()

#=== Error analysis (based on reference true value) ===
real_left = -922.00
real_up = 1799.00
real_right = 521.00

err_left = abs(real_left - C_left)
err_up = abs(real_up - C_Up)
err_right = abs(real_right - C_Right)

print("\n====== Error Analysis Table ======")
print(f"Left Cube  | Real: {real_left:.2f} mm | Predicted: {C_left:.2f} mm | Error: {err_left:.2f} mm")
print(f"Up Cube    | Real: {real_up:.2f} mm  | Predicted: {C_Up:.2f} mm  | Error: {err_up:.2f} mm")
print(f"Right Cube | Real: {real_right:.2f} mm | Predicted: {C_Right:.2f} mm | Error: {err_right:.2f} mm")

cv2.imwrite("debug_mask.png", img_bin * 255)
