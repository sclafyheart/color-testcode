import cv2
import numpy as np
import time
from picamera2 import Picamera2

# ---------------------------
# CAMERA SETUP
# ---------------------------
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "BGR888", "size": (640, 480)}
))
picam2.start()
time.sleep(1)

# ---------------------------
# HSV COLOR RANGES 
# ---------------------------
# Red (Handles the wrap-around 0-180)
lower_red1, upper_red1 = np.array([0, 120, 70]), np.array([10, 255, 255])
lower_red2, upper_red2 = np.array([170, 120, 70]), np.array([180, 255, 255])

# Green (Deepened to catch darker green)
lower_green, upper_green = np.array([35, 50, 20]), np.array([95, 255, 255])

# Blue
lower_blue, upper_blue = np.array([100, 150, 100]), np.array([120, 255, 255])

# Yellow (Tightened to ignore gray walls/floors)
lower_yellow, upper_yellow = np.array([20, 150, 150]), np.array([35, 255, 255])

# ---------------------------
# LOGIC FUNCTIONS
# ---------------------------
def identify_shape(contour):
    peri = cv2.arcLength(contour, True)
    # We use a smaller epsilon (0.02) to get a more detailed corner count
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    area = cv2.contourArea(contour)
    
    if peri == 0: return "Unknown"

    # Circularity: 1.0 is a perfect circle. 
    # We tighten this to 0.8 - 1.1 for Balls.
    circularity = (4 * np.pi * area) / (peri ** 2)
    
    # Aspect Ratio: Width vs Height
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h

    # 1. BALL LOGIC: Must be very round AND roughly square-shaped in dimensions
    if 0.8 < circularity < 1.1 and 0.8 < aspect_ratio < 1.2:
        return "Ball"
    
    # 2. BUCKET LOGIC: Must have 4 corners (approx) and LOWER circularity than a ball
    elif 4 <= len(approx) <= 8 and circularity < 0.8:
        return "Bucket"
        
    return "Unknown"

def get_color_masks(hsv):
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Clean up noise
    kernel = np.ones((5,5), np.uint8)
    combined_masks = {"Red": mask_red, "Green": mask_green, "Blue": mask_blue, "Yellow": mask_yellow}
    
    for color in combined_masks:
        combined_masks[color] = cv2.morphologyEx(combined_masks[color], cv2.MORPH_OPEN, kernel)
        combined_masks[color] = cv2.morphologyEx(combined_masks[color], cv2.MORPH_CLOSE, kernel)
        
    return combined_masks

# ---------------------------
# MAIN LOOP
# ---------------------------
print("Starting Detection... Press 'q' to quit.")
while True:
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    masks = get_color_masks(hsv)

    for color_name, mask in masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours:
            if cv2.contourArea(c) > 1500: # Ignore small wall artifacts
                x, y, w, h = cv2.boundingRect(c)
                shape = identify_shape(c)
                
                if shape != "Unknown":
                    # Draw visual feedback
                    color_bgr = (0, 255, 0) # Green box for all
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 2)
                    cv2.putText(frame, f"{color_name} {shape}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Ball and Bucket Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
