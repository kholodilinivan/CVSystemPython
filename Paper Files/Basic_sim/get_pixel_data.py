import cv2
import numpy as np

# Load image
image = cv2.imread('image.jpg')
image_display = image.copy()
selected_points = []

def mouse_callback(event, x, y, flags, param):
    global image_display
    
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_points.append((x, y))
        redraw_image()
        
    elif event == cv2.EVENT_RBUTTONDOWN:
        if selected_points:
            selected_points.pop()
            redraw_image()

def redraw_image():
    global image_display
    image_display = image.copy()
    
    for i, (x, y) in enumerate(selected_points):
        # Draw prominent point
        cv2.circle(image_display, (x, y), 8, (0, 0, 255), -1)
        cv2.circle(image_display, (x, y), 10, (255, 255, 255), 3)
        
        # Create coordinate text with point number
        coord_text = f'{i+1}:({x},{y})'
        
        # Use larger font for better visibility
        font_scale = 0.8
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            coord_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        
        # Position text
        text_x = x + 15
        text_y = y - 15
        
        # Boundary checking
        if text_x + text_width > image_display.shape[1]:
            text_x = x - text_width - 15
        if text_y - text_height < 10:
            text_y = y + text_height + 15
        
        # Draw solid black background with good padding
        bg_padding = 6
        bg_x1 = text_x - bg_padding
        bg_y1 = text_y - text_height - bg_padding
        bg_x2 = text_x + text_width + bg_padding
        bg_y2 = text_y + bg_padding
        
        # Ensure background stays within image bounds
        bg_x1 = max(0, bg_x1)
        bg_y1 = max(0, bg_y1)
        bg_x2 = min(image_display.shape[1], bg_x2)
        bg_y2 = min(image_display.shape[0], bg_y2)
        
        # Draw the background
        cv2.rectangle(image_display, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        
        # Optional: Add a white border for extra visibility
        cv2.rectangle(image_display, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), 1)
        
        # Draw text
        cv2.putText(image_display, coord_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    cv2.imshow('Image Inspector', image_display)

# Setup with larger initial window
cv2.namedWindow('Image Inspector', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image Inspector', 1000, 800)
cv2.imshow('Image Inspector', image_display)
cv2.setMouseCallback('Image Inspector', mouse_callback)

print("Click: Add point | Right-click: Remove last | 'c': Clear all | 'q': Quit")

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        selected_points.clear()
        redraw_image()

cv2.destroyAllWindows()