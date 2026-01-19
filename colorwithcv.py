import cv2
import numpy as np
import serial
import time

# ---------------------------
# SERIAL SETUP (PICO W)
# ---------------------------
ser = serial.Serial("COM3", 115200, timeout=1)  # Change COM port as needed

# ---------------------------
# CAMERA SETUP
# ---------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()
print("Press 'q' to quit")

# ---------------------------
# HSV COLOR RANGES
# ---------------------------
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

lower_green = np.array([36, 100, 100])
upper_green = np.array([86, 255, 255])

lower_blue = np.array([100, 150, 100])
upper_blue = np.array([120, 255, 255])

lower_yellow = np.array([18, 50, 150])
upper_yellow = np.array([35, 255, 255])

# ---------------------------
# HELPER FUNCTION
# ---------------------------
def largest_blob_size(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    return max(cv2.contourArea(c) for c in contours)

# ---------------------------
# VARIABLES
# ---------------------------
last_color = None
fps_limit = 1 / 30  # 30 FPS

# ---------------------------
# MAIN LOOP
# ---------------------------
while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2     # compute frame center
    box_size = 100              # size of initial ROI around cube

    # Initial ROI coordinates in the full frame
    x1 = max(cx - box_size, 0)
    y1 = max(cy - box_size, 0)
    x2 = min(cx + box_size, w)
    y2 = min(cy + box_size, h)

    # Crop ROI from frame
    roi = frame[y1:y2, x1:x2]

    # Step 2: convert to HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Step 3: saturation + brightness mask
    lower_s = 150
    lower_v = 150
    sat_val_mask = cv2.inRange(hsv, np.array([0, lower_s, lower_v]), np.array([180, 255, 255]))

    kernel = np.ones((3,3), np.uint8)
    sat_val_mask = cv2.morphologyEx(sat_val_mask, cv2.MORPH_OPEN, kernel)

    # Step 4: find largest contour in mask
    contours, _ = cv2.findContours(sat_val_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cube_contour = None
    for c in contours:
        area = cv2.contourArea(c)
        x, y, w_box, h_box = cv2.boundingRect(c)
        
        # Only consider contours near center
        if 3000 < area < 20000 and abs((x + w_box//2) - cx) < 100 and abs((y + h_box//2) - cy) < 100:
            cube_contour = c
            break

    # Step 5: adjust ROI if cube found
    if cube_contour is not None:
        x, y, w_box, h_box = cv2.boundingRect(cube_contour)
        padding = 20  # extra pixels around cube
        x1 = max(x1 + x - padding, 0)
        y1 = max(y1 + y - padding, 0)
        x2 = min(x1 + w_box + 2*padding, w)
        y2 = min(y1 + h_box + 2*padding, h)
        roi = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    else:
        # fallback to previous ROI or center box
        roi = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # ---------------------------
    # COLOR MASKS
    # ---------------------------
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Morphology to remove noise
    kernel = np.ones((3,3), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)

    # ---------------------------
    # DETECT STRONGEST COLOR
    # ---------------------------
    counts = {
        "Red": largest_blob_size(mask_red),
        "Green": largest_blob_size(mask_green),
        "Blue": largest_blob_size(mask_blue),
        "Yellow": largest_blob_size(mask_yellow)
    }
    detected_color = max(counts, key=counts.get)
    if counts[detected_color] < 200:  # ignore tiny blobs
        detected_color = "None"

    # ---------------------------
    # GRAYSCALE BACKGROUND + CUBE COLOR
    # ---------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    mask_cube = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask_cube[y1:y2, x1:x2] = 255
    color_only = cv2.bitwise_and(frame, frame, mask=mask_cube)
    final = cv2.add(color_only, cv2.bitwise_and(gray_bgr, gray_bgr, mask=cv2.bitwise_not(mask_cube)))

    # Draw bounding box
    cv2.rectangle(final, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Display detected color
    cv2.putText(final, f"Detected: {detected_color}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # ---------------------------
    # SEND COLOR TO PICO ONLY IF CHANGED
    # ---------------------------
    if detected_color != "None" and detected_color != last_color:
        ser.write((detected_color.upper() + "\n").encode())
        last_color = detected_color

    # Show frame
    cv2.imshow("Rubik's Cube Color Detection", final)

    # ---------------------------
    # QUIT
    # ---------------------------
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # ---------------------------
    # LIMIT FPS
    # ---------------------------
    elapsed = time.time() - start_time
    if elapsed < fps_limit:
        time.sleep(fps_limit - elapsed)

# ---------------------------
# CLEANUP
# ---------------------------
cap.release()
cv2.destroyAllWindows()
