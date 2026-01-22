from picamera2 import Picamera2
import cv2
import numpy as np
import time

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
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

lower_green = np.array([35, 50, 30])
upper_green = np.array([90, 255, 180])

lower_blue = np.array([100, 150, 100])
upper_blue = np.array([120, 255, 255])

lower_yellow = np.array([18, 50, 150])
upper_yellow = np.array([35, 255, 255])

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def find_blobs(mask, min_area=500):
    """
    Returns list of bounding boxes for blobs above min_area.
    Each box: (x, y, w, h)
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((x, y, w, h, area))
    return boxes

def get_color_masks(hsv):
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | \
               cv2.inRange(hsv, lower_red2, upper_red2)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Green: combine dark + bright
    mask_green_dark = cv2.inRange(hsv, np.array([36, 50, 30]), np.array([86, 255, 180]))
    mask_green_bright = cv2.inRange(hsv, np.array([36, 70, 50]), np.array([86, 255, 255]))
    mask_green = mask_green_dark | mask_green_bright

    # morphology to remove noise
    kernel = np.ones((5,5), np.uint8)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
    return {
        "Red": mask_red,
        "Green": mask_green,
        "Blue": mask_blue,
        "Yellow": mask_yellow
    }

# ---------------------------
# MAIN LOOP
# ---------------------------
while True:
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Fix colors
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    masks = get_color_masks(hsv)

    detected_colors = {}  # Store blobs for each color

    # ---------------------------
    # DETECT BLOBS AND DRAW BOXES
    # ---------------------------
    for color, mask in masks.items():
        boxes = find_blobs(mask)
        detected_colors[color] = boxes
        for (x, y, w, h, area) in boxes:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
            cv2.putText(frame, f"{color}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # ---------------------------
    # SHOW FRAME
    # ---------------------------
    cv2.imshow("Multi-Color Detection", frame)

    # ---------------------------
    # QUIT
    # ---------------------------
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

