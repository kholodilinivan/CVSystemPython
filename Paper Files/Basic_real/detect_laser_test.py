
import cv2
import numpy as np
from PIL import Image

red = np.uint8([[[255,0,0 ]]]) # r g b
hsv_red = cv2.cvtColor(red,cv2.COLOR_BGR2HSV) # convert to HSV
print (hsv_red)

image = cv2.imread('image1_.jpg') # load test image

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([0, 200, 200]) # lower red intensity in HSV
upper = np.array([255, 255, 255]) # higher red intensity in HSV
mask = cv2.inRange(hsv, lower, upper)
result = cv2.bitwise_and(image, image, mask=mask)

img = result

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Apply basic thresholding
_, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

cv2.imshow("Binary", binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()