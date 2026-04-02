#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np


class MoveToBucketNode(Node):
    def __init__(self):
        super().__init__('move_to_bucket_node')

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Red HSV ranges
        self.lower_red1 = np.array([0, 120, 70])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 120, 70])
        self.upper_red2 = np.array([180, 255, 255])

        self.kernel = np.ones((5, 5), np.uint8)

        # 🧠 For smoothing centroid (reduces wobble)
        self.prev_cx = None

        self.get_logger().info("MoveToBucketNode running")

    def get_color_mask(self, hsv):
        mask = cv2.inRange(hsv, self.lower_red1, self.upper_red1) | \
               cv2.inRange(hsv, self.lower_red2, self.upper_red2)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        return mask

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return

        # 🔄 FIX: flip upside-down camera
        frame = cv2.flip(frame, -1)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = self.get_color_mask(hsv)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        twist = Twist()

        height, width, _ = frame.shape

        # 🔍 SEARCH MODE
        if not contours:
            twist.angular.z = 0.4
            self.cmd_pub.publish(twist)

            cv2.imshow("MoveToBucket", frame)
            cv2.waitKey(1)
            return

        # Get largest contour
        c = max(contours, key=cv2.contourArea)

        if cv2.contourArea(c) < 1500:
            twist.angular.z = 0.4
            self.cmd_pub.publish(twist)
            return

        # 📍 Centroid
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            return

        # 🧠 Smooth centroid (reduces jitter)
        if self.prev_cx is None:
            smoothed_cx = cx
        else:
            smoothed_cx = int(0.7 * self.prev_cx + 0.3 * cx)

        self.prev_cx = smoothed_cx
        error_x = smoothed_cx - width // 2

        # Bounding box (for width-based stopping)
        x, y, w, h = cv2.boundingRect(c)

        # 🎯 Draw visuals
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (smoothed_cx, cy), 5, (0, 0, 255), -1)
        cv2.line(frame, (width // 2, 0), (width // 2, height), (255, 0, 0), 2)

        # 🔄 Smooth steering
        twist.angular.z = -0.002 * error_x
        twist.angular.z = np.clip(twist.angular.z, -0.4, 0.4)

        # 🧊 Dead zone
        if abs(error_x) < 20:
            twist.angular.z = 0.0

        # 📏 Width-based stopping
        target_width = int(width * 0.6)

        # 🚗 Forward control
        if w < target_width:
            distance_scale = 1 - (w / target_width)

            center_error = abs(error_x)
            alignment_scale = 1 - min(center_error / (width / 2), 1)

            speed = 0.15 * distance_scale * alignment_scale
            twist.linear.x = max(0.05, speed)

        else:
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        self.cmd_pub.publish(twist)

        cv2.imshow("MoveToBucket", frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = MoveToBucketNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()