#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class ColorDetectionNode(Node):
    def __init__(self):
        super().__init__('color_detection_node')

        # Subscribe to your camera topic (adjust if needed)
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',  # Change this to your actual camera topic
            self.image_callback,
            10
        )

        self.bridge = CvBridge()
        self.get_logger().info("Color Detection Node Started!")

        # HSV color ranges
        self.lower_red1, self.upper_red1 = np.array([0, 120, 70]), np.array([10, 255, 255])
        self.lower_red2, self.upper_red2 = np.array([170, 120, 70]), np.array([180, 255, 255])
        self.lower_green, self.upper_green = np.array([35, 50, 20]), np.array([95, 255, 255])
        self.lower_blue, self.upper_blue = np.array([100, 150, 100]), np.array([120, 255, 255])
        self.lower_yellow, self.upper_yellow = np.array([20, 150, 150]), np.array([35, 255, 255])

        self.kernel = np.ones((5,5), np.uint8)

    def identify_shape(self, contour):
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        area = cv2.contourArea(contour)

        if peri == 0:
            return "Unknown"

        circularity = (4 * np.pi * area) / (peri ** 2)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h

        if 0.8 < circularity < 1.1 and 0.8 < aspect_ratio < 1.2:
            return "Ball"
        elif 4 <= len(approx) <= 8 and circularity < 0.8:
            return "Bucket"
        return "Unknown"

    def get_color_masks(self, hsv):
        mask_red = cv2.inRange(hsv, self.lower_red1, self.upper_red1) | cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask_green = cv2.inRange(hsv, self.lower_green, self.upper_green)
        mask_blue = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        mask_yellow = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)

        combined_masks = {"Red": mask_red, "Green": mask_green, "Blue": mask_blue, "Yellow": mask_yellow}
        for color in combined_masks:
            combined_masks[color] = cv2.morphologyEx(combined_masks[color], cv2.MORPH_OPEN, self.kernel)
            combined_masks[color] = cv2.morphologyEx(combined_masks[color], cv2.MORPH_CLOSE, self.kernel)

        return combined_masks

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        masks = self.get_color_masks(hsv)

        for color_name, mask in masks.items():
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) > 1500:
                    x, y, w, h = cv2.boundingRect(c)
                    shape = self.identify_shape(c)
                    if shape != "Unknown":
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, f"{color_name} {shape}", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Ball and Bucket Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = ColorDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()