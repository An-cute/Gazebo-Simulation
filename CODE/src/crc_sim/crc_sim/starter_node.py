import math
import time
from typing import Optional

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan


class PreciseLaneFollower(Node):

    def __init__(self):
        super().__init__('precise_lane_follower')

        # =========================================================
        # ROS 2
        # =========================================================
        self.bridge = CvBridge()

        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.create_subscription(
            Image,
            '/camera/image_raw',
            self._cb_image,
            qos_profile_sensor_data
        )

        self.create_subscription(
            LaserScan,
            '/scan',
            self._cb_scan,
            qos_profile_sensor_data
        )

        self.create_subscription(
            Odometry,
            '/odom',
            self._cb_odom,
            10
        )

        # =========================================================
        # CAMERA / SENSOR STATE
        # =========================================================
        self.image: Optional[np.ndarray] = None
        self.scan: Optional[LaserScan] = None

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        # =========================================================
        # CLAHE
        # =========================================================
        self.clahe = cv2.createCLAHE(
            clipLimit=3.0,
            tileGridSize=(8, 8)
        )

        # =========================================================
        # LANE FOLLOWING
        # =========================================================
        self.dist_to_right_line = 135.0
        self.last_target_x: Optional[float] = None
        self.kp = 0.0075
        self.max_angular = 1.85
        self.base_speed = 0.22

        # =========================================================
        # TRAFFIC LIGHT / STOP SIGN
        # =========================================================
        self.object_state = "CLEAR"
        self.required_confirm_frames = 4

        self.red_count = 0
        self.yellow_count = 0
        self.stop_sign_count = 0

        self.stop_sign_timer: Optional[float] = None
        self.stop_sign_cooldown = 0.0
        self.stop_duration = 2.2

        self.red_light_confirmed = False
        self.yellow_light_confirmed = False

        # =========================================================
        # TIMER
        # =========================================================
        self.timer = self.create_timer(
            0.05,
            self.control
        )

        self.get_logger().info(
            "PreciseLaneFollower started"
        )

    # =============================================================
    # IMAGE CALLBACK
    # =============================================================

    def _cb_image(self, msg: Image):

        try:
            self.image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )

        except Exception as e:
            self.get_logger().error(
                f"Lỗi CvBridge: {e}"
            )

    # =============================================================
    # LIDAR CALLBACK
    # =============================================================

    def _cb_scan(self, msg: LaserScan):
        self.scan = msg

    # =============================================================
    # ODOM CALLBACK
    # =============================================================

    def _cb_odom(self, msg: Odometry):

        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation

        siny_cosp = 2.0 * (
            q.w * q.z +
            q.x * q.y
        )

        cosy_cosp = 1.0 - 2.0 * (
            q.y * q.y +
            q.z * q.z
        )

        self.yaw = math.atan2(
            siny_cosp,
            cosy_cosp
        )

    # =============================================================
    # LIDAR
    # =============================================================

    def get_min_range(
        self,
        start_deg: float,
        end_deg: float
    ) -> float:

        if (
            self.scan is None
            or not self.scan.ranges
        ):
            return float('inf')

        min_val = float('inf')

        total_rays = len(
            self.scan.ranges
        )

        for deg in np.linspace(
            start_deg,
            end_deg,
            num=12
        ):

            idx = int(
                (deg % 360)
                * total_rays
                / 360.0
            )

            if 0 <= idx < total_rays:

                r = self.scan.ranges[idx]

                if (
                    not math.isnan(r)
                    and not math.isinf(r)
                    and r > 0.04
                ):

                    min_val = min(
                        min_val,
                        r
                    )

        return min_val

    # =============================================================
    # DRIVE
    # =============================================================

    def drive(
        self,
        v: float,
        w: float
    ):

        msg = Twist()

        msg.linear.x = float(v)
        msg.angular.z = float(w)

        self.cmd_pub.publish(msg)

    # =============================================================
    # STOP
    # =============================================================

    def stop(self):
        self.drive(0.0, 0.0)

    # =============================================================
    # TRAFFIC LIGHT + STOP SIGN DETECTION
    # =============================================================

    def detect_traffic_light_and_signs(
        self,
        full_image: np.ndarray
    ) -> str:

        h, w, _ = full_image.shape

        upper_roi = full_image[
            0:int(h * 0.55),
            :
        ]

        hsv = cv2.cvtColor(
            upper_roi,
            cv2.COLOR_BGR2HSV
        )

        # RED
        mask_red1 = cv2.inRange(
            hsv,
            np.array([0, 120, 100]),
            np.array([10, 255, 255])
        )

        mask_red2 = cv2.inRange(
            hsv,
            np.array([160, 120, 100]),
            np.array([180, 255, 255])
        )

        mask_red = cv2.bitwise_or(
            mask_red1,
            mask_red2
        )

        # YELLOW
        mask_yellow = cv2.inRange(
            hsv,
            np.array([18, 120, 120]),
            np.array([34, 255, 255])
        )

        red_kernel = np.ones(
            (3, 3),
            np.uint8
        )

        mask_red_clean = cv2.morphologyEx(
            mask_red,
            cv2.MORPH_CLOSE,
            red_kernel
        )

        # SLOPE SIGN
        red_contours, _ = cv2.findContours(
            mask_red_clean,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for c in red_contours:

            area = cv2.contourArea(c)

            if area < 150:
                continue

            perimeter = cv2.arcLength(
                c,
                True
            )

            if perimeter <= 0:
                continue

            hull = cv2.convexHull(c)

            hull_perimeter = cv2.arcLength(
                hull,
                True
            )

            if hull_perimeter <= 0:
                continue

            triangle = cv2.approxPolyDP(
                hull,
                0.04 * hull_perimeter,
                True
            )

            sx, sy, sw, sh = cv2.boundingRect(
                hull
            )

            ratio = (
                float(sw)
                / max(sh, 1)
            )

            if (
                len(triangle) == 3
                and 0.60 < ratio < 1.50
            ):

                if (
                    sx < 5
                    or sy < 5
                    or sx + sw > w - 5
                ):
                    continue

                if sw < 20 or sh < 20:
                    continue

                triangle_area = cv2.contourArea(
                    triangle
                )

                if triangle_area <= 0:
                    continue

                hull_area = cv2.contourArea(
                    hull
                )

                if hull_area <= 0:
                    continue

                solidity = (
                    area
                    / hull_area
                )

                if solidity > 0.25:
                    return "SLOPE_SIGN"

        # TRAFFIC LIGHT
        gray_upper = cv2.cvtColor(
            upper_roi,
            cv2.COLOR_BGR2GRAY
        )

        _, dark_box_mask = cv2.threshold(
            gray_upper,
            45,
            255,
            cv2.THRESH_BINARY_INV
        )

        dark_box_mask = cv2.morphologyEx(
            dark_box_mask,
            cv2.MORPH_CLOSE,
            np.ones((5, 5), np.uint8)
        )

        box_contours, _ = cv2.findContours(
            dark_box_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for c in box_contours:

            area = cv2.contourArea(c)

            if area < 300:
                continue

            bx, by, bw, bh = cv2.boundingRect(c)

            if (
                bx < 0
                or by < 0
                or bx + bw > w
                or by + bh > int(h * 0.55)
            ):
                continue

            aspect_ratio = (
                float(bh)
                / max(bw, 1)
            )

            if not (
                1.5
                < aspect_ratio
                < 4.0
            ):
                continue

            if bw < 8 or bh < 20:
                continue

            box_red = mask_red_clean[
                by:by + bh,
                bx:bx + bw
            ]

            red_contours_box, _ = cv2.findContours(
                box_red,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            red_light_found = False

            for rc in red_contours_box:

                red_area = cv2.contourArea(rc)

                if red_area < 20:
                    continue

                rx, ry, rw, rh = cv2.boundingRect(rc)

                red_ratio = (
                    float(rw)
                    / max(rh, 1)
                )

                if not (
                    0.65
                    < red_ratio
                    < 1.35
                ):
                    continue

                red_perimeter = cv2.arcLength(
                    rc,
                    True
                )

                if red_perimeter <= 0:
                    continue

                circularity = (
                    4.0
                    * math.pi
                    * red_area
                    /
                    (
                        red_perimeter
                        * red_perimeter
                    )
                )

                if circularity < 0.45:
                    continue

                red_area_ratio = (
                    float(red_area)
                    / float(max(bw * bh, 1))
                )

                if red_area_ratio > 0.20:
                    continue

                red_light_found = True
                break

            if red_light_found:
                return "RED_LIGHT"

            box_yellow = mask_yellow[
                by:by + bh,
                bx:bx + bw
            ]

            yellow_contours_box, _ = cv2.findContours(
                box_yellow,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for yc in yellow_contours_box:

                yellow_area = cv2.contourArea(yc)

                if yellow_area < 20:
                    continue

                yx, yy, yw, yh = cv2.boundingRect(yc)

                yellow_ratio = (
                    float(yw)
                    / max(yh, 1)
                )

                if not (
                    0.65
                    < yellow_ratio
                    < 1.35
                ):
                    continue

                yellow_perimeter = cv2.arcLength(
                    yc,
                    True
                )

                if yellow_perimeter <= 0:
                    continue

                yellow_circularity = (
                    4.0
                    * math.pi
                    * yellow_area
                    /
                    (
                        yellow_perimeter
                        * yellow_perimeter
                    )
                )

                if yellow_circularity < 0.45:
                    continue

                yellow_area_ratio = (
                    float(yellow_area)
                    / float(max(bw * bh, 1))
                )

                if yellow_area_ratio > 0.20:
                    continue

                return "YELLOW_LIGHT"

        # STOP SIGN
        for c in red_contours:

            area = cv2.contourArea(c)

            if area < 500:
                continue

            sx, sy, sw, sh = cv2.boundingRect(c)

            ratio = (
                float(sw)
                / max(sh, 1)
            )

            if not (
                0.70
                < ratio
                < 1.40
            ):
                continue

            if (
                sx < 10
                or sx + sw > w - 10
            ):
                continue

            perimeter = cv2.arcLength(
                c,
                True
            )

            if perimeter > 0:

                hull = cv2.convexHull(c)

                hull_perimeter = cv2.arcLength(
                    hull,
                    True
                )

                if hull_perimeter > 0:

                    approx = cv2.approxPolyDP(
                        hull,
                        0.04 * hull_perimeter,
                        True
                    )

                    if len(approx) == 3:
                        continue

            return "STOP_SIGN"

        return "CLEAR"

    # =============================================================
    # OBJECT CONFIRMATION
    # =============================================================

    def update_object_confirmation(
        self,
        detected: str
    ) -> str:

        if detected == "RED_LIGHT":
            self.red_count += 1
        else:
            self.red_count = max(
                0,
                self.red_count - 1
            )

        if detected == "YELLOW_LIGHT":
            self.yellow_count += 1
        else:
            self.yellow_count = max(
                0,
                self.yellow_count - 1
            )

        if detected == "STOP_SIGN":
            self.stop_sign_count += 1
        else:
            self.stop_sign_count = max(
                0,
                self.stop_sign_count - 1
            )

        if (
            self.red_count
            >= self.required_confirm_frames
        ):
            return "RED_LIGHT"

        if (
            self.yellow_count
            >= self.required_confirm_frames
        ):
            return "YELLOW_LIGHT"

        if (
            self.stop_sign_count
            >= self.required_confirm_frames
        ):
            return "STOP_SIGN"

        return "CLEAR"

    # =============================================================
    # TRAFFIC CONTROL
    # =============================================================

    def handle_traffic_control(
        self,
        status: str,
        current_time: float
    ) -> bool:

        if self.stop_sign_timer is not None:

            elapsed = (
                current_time
                - self.stop_sign_timer
            )

            if elapsed < self.stop_duration:

                self.stop()

                return True

            self.stop_sign_timer = None

            self.stop_sign_cooldown = (
                current_time + 8.0
            )

            self.get_logger().info(
                "STOP completed - continue"
            )

        if status == "RED_LIGHT":

            self.get_logger().info(
                "RED LIGHT -> STOP"
            )

            self.stop()

            return True

        if status == "YELLOW_LIGHT":

            self.get_logger().info(
                "YELLOW LIGHT -> STOP"
            )

            self.stop()

            return True

        if (
            status == "STOP_SIGN"
            and current_time
            > self.stop_sign_cooldown
        ):

            self.get_logger().info(
                "STOP SIGN confirmed -> STOP 2.2s"
            )

            self.stop_sign_timer = current_time

            self.stop()

            return True

        return False

    # =============================================================
    # LANE FOLLOWING - GIỮ NGUYÊN
    # =============================================================

    def lane_following(self):

        h, w, _ = self.image.shape

        camera_center = w / 2.0

        roi = self.image[
            int(h * 0.35):h,
            0:w
        ]

        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY
        )

        mean_val = float(
            np.mean(gray)
        )

        if mean_val < 85.0:

            enhanced = self.clahe.apply(
                gray
            )

            _, thresh = cv2.threshold(
                enhanced,
                210,
                255,
                cv2.THRESH_BINARY
            )

            thresh[:, :18] = 0
            thresh[:, -18:] = 0

        else:

            _, thresh = cv2.threshold(
                gray,
                195,
                255,
                cv2.THRESH_BINARY
            )

            thresh[:, :10] = 0
            thresh[:, -10:] = 0

        mid_x = int(camera_center)

        thresh_right = thresh[
            :,
            mid_x:w
        ]

        thresh_left = thresh[
            :,
            0:mid_x
        ]

        m_right = cv2.moments(
            thresh_right
        )

        m_left = cv2.moments(
            thresh_left
        )

        if m_right['m00'] > 30:

            right_line_x = (
                float(
                    m_right['m10']
                    / m_right['m00']
                )
                + mid_x
            )

            target_x = (
                right_line_x
                - self.dist_to_right_line
            )

            self.last_target_x = target_x

        elif m_left['m00'] > 30:

            left_line_x = float(
                m_left['m10']
                / m_left['m00']
            )

            target_x = (
                left_line_x
                + self.dist_to_right_line
            )

            self.last_target_x = target_x

        else:

            target_x = (
                self.last_target_x
                if self.last_target_x is not None
                else camera_center
            )

        error = (
            camera_center
            - target_x
        )

        angular_z = (
            self.kp
            * error
        )

        dist_right_wall = (
            self.get_min_range(
                280.0,
                320.0
            )
        )

        dist_left_wall = (
            self.get_min_range(
                40.0,
                80.0
            )
        )

        if dist_right_wall < 0.20:

            angular_z += (
                0.35
                * (
                    0.20
                    - dist_right_wall
                )
                / 0.20
            )

        elif dist_left_wall < 0.20:

            angular_z -= (
                0.35
                * (
                    0.20
                    - dist_left_wall
                )
                / 0.20
            )

        angular_z = max(
            -self.max_angular,
            min(
                self.max_angular,
                angular_z
            )
        )

        linear_x = max(
            0.12,
            self.base_speed
            * (
                1.0
                - min(
                    abs(error) / 250.0,
                    0.45
                )
            )
        )

        self.drive(
            linear_x,
            angular_z
        )

    # =============================================================
    # ORIGINAL CONTROL
    # =============================================================

    def control(self):

        if self.image is None:
            return

        current_time = time.monotonic()

        detected = (
            self.detect_traffic_light_and_signs(
                self.image
            )
        )

        status = (
            self.update_object_confirmation(
                detected
            )
        )

        traffic_handled = (
            self.handle_traffic_control(
                status,
                current_time
            )
        )

        if traffic_handled:
            return

        self.lane_following()


# =================================================================
# HW SIGN
# =================================================================

def _hw_detect_sign(self, image):

    if image is None:
        return False, None

    h, w, _ = image.shape

    roi = image[
        0:int(h * 0.70),
        :
    ]

    hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2HSV
    )

    green_mask = cv2.inRange(
        hsv,
        np.array([45, 100, 80]),
        np.array([90, 255, 255])
    )

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    green_mask = cv2.morphologyEx(
        green_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    green_mask = cv2.morphologyEx(
        green_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    contours, _ = cv2.findContours(
        green_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 500:
            continue

        x, y, cw, ch = cv2.boundingRect(
            contour
        )

        ratio = (
            float(cw)
            / max(ch, 1)
        )

        if not (
            0.70
            < ratio
            < 1.35
        ):
            continue

        if cw < 25 or ch < 25:
            continue

        x1 = max(
            0,
            x + int(cw * 0.15)
        )

        y1 = max(
            0,
            y + int(ch * 0.15)
        )

        x2 = min(
            roi.shape[1],
            x + int(cw * 0.85)
        )

        y2 = min(
            roi.shape[0],
            y + int(ch * 0.85)
        )

        if x2 <= x1 or y2 <= y1:
            continue

        sign_hsv = hsv[
            y1:y2,
            x1:x2
        ]

        white_mask = cv2.inRange(
            sign_hsv,
            np.array([0, 0, 150]),
            np.array([180, 100, 255])
        )

        white_ratio = (
            float(
                cv2.countNonZero(
                    white_mask
                )
            )
            / float(
                max(
                    white_mask.size,
                    1
                )
            )
        )

        if white_ratio < 0.08:
            continue

        if white_ratio > 0.65:
            continue

        center_x = (
            x + cw / 2.0
        )

        center_y = (
            y + ch / 2.0
        )

        return True, (
            float(center_x),
            float(center_y),
            float(cw),
            float(ch)
        )

    return False, None


# =================================================================
# HW LEFT LANE CHANGE
# =================================================================

def _hw_lane_change_control(self):

    now = time.monotonic()

    if not getattr(
        self,
        "hw_lane_change_active",
        False
    ):
        return False

    elapsed = (
        now
        - self.hw_lane_change_start
    )

    duration = self.hw_lane_change_duration

    if elapsed >= duration:

        self.hw_lane_change_active = False
        self.hw_lane_change_done = True

        self.hw_lane_change_cooldown = (
            now + 8.0
        )

        self.hw_last_angular = 0.0

        self.get_logger().info(
            "HW -> LEFT LANE CHANGE COMPLETED"
        )

        return False

    if elapsed < 1.5:

        progress = (
            elapsed
            / 1.5
        )

        target_angular = (
            math.radians(20.0)
            * progress
        )

    elif elapsed < 2.5:

        target_angular = math.radians(
            20.0
        )

    else:

        progress = (
            elapsed - 2.5
        ) / (
            duration - 2.5
        )

        progress = max(
            0.0,
            min(
                1.0,
                progress
            )
        )

        target_angular = (
            math.radians(20.0)
            * (
                1.0 - progress
            )
        )

    last_angular = getattr(
        self,
        "hw_last_angular",
        0.0
    )

    max_step = 0.012

    if target_angular > (
        last_angular + max_step
    ):

        angular = (
            last_angular
            + max_step
        )

    elif target_angular < (
        last_angular - max_step
    ):

        angular = (
            last_angular
            - max_step
        )

    else:

        angular = target_angular

    self.hw_last_angular = angular

    self.drive(
        0.145,
        angular
    )

    return True


# =================================================================
# HW INIT WRAPPER
# =================================================================

_old_precise_init = PreciseLaneFollower.__init__


def _new_precise_init(
    self,
    *args,
    **kwargs
):

    _old_precise_init(
        self,
        *args,
        **kwargs
    )

    self.hw_count = 0
    self.hw_required_frames = 2
    self.hw_lane_change_active = False
    self.hw_lane_change_done = False
    self.hw_lane_change_start = 0.0
    self.hw_lane_change_duration = 3.8
    self.hw_lane_change_cooldown = 0.0
    self.hw_last_angular = 0.0

    self.get_logger().info(
        "HW LEFT-LANE CHANGE ADD-ON ENABLED"
    )


PreciseLaneFollower.__init__ = _new_precise_init


# =================================================================
# HW CONTROL WRAPPER
# =================================================================

_old_precise_control = PreciseLaneFollower.control


def _new_precise_control(self):

    if self.image is None:
        return

    current_time = time.monotonic()

    if getattr(
        self,
        "hw_lane_change_active",
        False
    ):

        if _hw_lane_change_control(self):
            return

    hw_found, hw_info = _hw_detect_sign(
        self,
        self.image
    )

    if hw_found:

        self.hw_count += 1

    else:

        self.hw_count = max(
            0,
            self.hw_count - 1
        )

    if (
        self.hw_count
        >= self.hw_required_frames
        and not self.hw_lane_change_done
        and current_time
        > self.hw_lane_change_cooldown
    ):

        self.hw_lane_change_active = True

        self.hw_lane_change_start = (
            current_time
        )

        self.hw_last_angular = 0.0

        self.get_logger().info(
            "HW SIGN DETECTED -> "
            "GENTLE LEFT LANE CHANGE"
        )

        _hw_lane_change_control(self)

        return

    _old_precise_control(self)


PreciseLaneFollower.control = _new_precise_control


# =================================================================
# HW LEFT-LANE FOLLOWING
# =================================================================

def _hw_left_lane_following(self):

    if self.image is None:
        return

    h, w, _ = self.image.shape

    camera_center = w / 2.0

    roi = self.image[
        int(h * 0.35):h,
        0:w
    ]

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    mean_val = float(
        np.mean(gray)
    )

    if mean_val < 85.0:

        enhanced = self.clahe.apply(
            gray
        )

        _, thresh = cv2.threshold(
            enhanced,
            210,
            255,
            cv2.THRESH_BINARY
        )

        thresh[:, :18] = 0
        thresh[:, -18:] = 0

    else:

        _, thresh = cv2.threshold(
            gray,
            195,
            255,
            cv2.THRESH_BINARY
        )

        thresh[:, :10] = 0
        thresh[:, -10:] = 0

    # Loại block nhỏ / vạch ngang để tránh target nhảy.
    filtered_thresh = np.zeros_like(
        thresh
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 20:
            continue

        x, y, cw, ch = cv2.boundingRect(
            contour
        )

        if cw <= 0 or ch <= 0:
            continue

        aspect = (
            float(cw)
            / float(ch)
        )

        horizontal_marking = (
            ch <= 30
            and cw >= 18
            and aspect >= 0.75
        )

        small_square_marking = (
            cw <= 45
            and ch <= 45
            and 0.65 <= aspect <= 1.55
        )

        long_horizontal_marking = (
            cw >= 45
            and aspect >= 2.0
            and ch <= 40
        )

        if (
            horizontal_marking
            or small_square_marking
            or long_horizontal_marking
        ):
            continue

        cv2.drawContours(
            filtered_thresh,
            [contour],
            -1,
            255,
            thickness=cv2.FILLED
        )

    mid_x = int(
        camera_center
    )

    thresh_left = filtered_thresh[
        :,
        0:mid_x
    ]

    m_left = cv2.moments(
        thresh_left
    )

    if m_left['m00'] > 30:

        left_line_x = float(
            m_left['m10']
            / m_left['m00']
        )

        target_x = (
            left_line_x
            + self.dist_to_right_line
        )

        self.hw_left_last_target_x = (
            target_x
        )

        self.last_target_x = target_x

    else:

        target_x = getattr(
            self,
            "hw_left_last_target_x",
            None
        )

        if target_x is None:

            target_x = (
                self.last_target_x
                if self.last_target_x is not None
                else camera_center
            )

    error = (
        camera_center
        - target_x
    )

    angular_z = (
        self.kp
        * error
    )

    dist_right_wall = (
        self.get_min_range(
            280.0,
            320.0
        )
    )

    dist_left_wall = (
        self.get_min_range(
            40.0,
            80.0
        )
    )

    if dist_right_wall < 0.20:

        angular_z += (
            0.35
            * (
                0.20
                - dist_right_wall
            )
            / 0.20
        )

    elif dist_left_wall < 0.20:

        angular_z -= (
            0.35
            * (
                0.20
                - dist_left_wall
            )
            / 0.20
        )

    # 20 độ/giây = 0.349 rad/s cho angular.z
    hw_max_angular = math.radians(
        20.0
    )

    angular_z = max(
        -hw_max_angular,
        min(
            hw_max_angular,
            angular_z
        )
    )

    linear_x = max(
        0.12,
        self.base_speed
        * (
            1.0
            - min(
                abs(error) / 250.0,
                0.45
            )
        )
    )

    linear_x = min(
        linear_x,
        0.18
    )

    self.drive(
        linear_x,
        angular_z
    )


# =================================================================
# GHI ĐÈ HW CONTROL ĐỂ SAU KHI CHUYỂN LÀN THÌ BÁM LINE TRÁI
# =================================================================

_previous_hw_control = PreciseLaneFollower.control


def _hw_final_control(self):

    if self.image is None:
        return

    current_time = time.monotonic()

    # -------------------------------------------------------------
    # TRAFFIC ƯU TIÊN
    # -------------------------------------------------------------

    detected = (
        self.detect_traffic_light_and_signs(
            self.image
        )
    )

    status = (
        self.update_object_confirmation(
            detected
        )
    )

    traffic_handled = (
        self.handle_traffic_control(
            status,
            current_time
        )
    )

    if traffic_handled:
        return

    # -------------------------------------------------------------
    # ĐANG CHUYỂN LÀN
    # -------------------------------------------------------------

    if getattr(
        self,
        "hw_lane_change_active",
        False
    ):

        _hw_lane_change_control(self)

        return

    # -------------------------------------------------------------
    # ĐÃ SANG LÀN TRÁI
    # -------------------------------------------------------------

    if getattr(
        self,
        "hw_left_lane_mode",
        False
    ):

        _hw_left_lane_following(self)

        return

    # -------------------------------------------------------------
    # PHÁT HIỆN HW
    # -------------------------------------------------------------

    hw_found, _ = _hw_detect_sign(
        self,
        self.image
    )

    if hw_found:

        self.hw_count += 1

    else:

        self.hw_count = max(
            0,
            self.hw_count - 1
        )

    if (
        self.hw_count
        >= self.hw_required_frames
        and not self.hw_lane_change_done
        and current_time
        > self.hw_lane_change_cooldown
    ):

        self.hw_lane_change_active = True

        self.hw_lane_change_start = (
            current_time
        )

        self.hw_last_angular = 0.0

        self.get_logger().info(
            "HW SIGN DETECTED -> "
            "LEFT LANE"
        )

        _hw_lane_change_control(self)

        return

    # -------------------------------------------------------------
    # CODE CŨ
    # -------------------------------------------------------------

    self.lane_following()


PreciseLaneFollower.control = _hw_final_control


# =================================================================
# WHITE DASHED CROSS LINE ADD-ON
#
# Đây là phần mới duy nhất dành cho vạch trắng sọc ngang.
#
# Không sửa lane_following() gốc.
# Không sửa HW.
# Không sửa traffic.
# =================================================================


def _detect_white_dashed_cross_line_addon(
    self,
    image
):

    if image is None:
        return False, None

    h, w, _ = image.shape

    # -------------------------------------------------------------
    # ROI
    # -------------------------------------------------------------

    y_start = int(h * 0.30)
    y_end = int(h * 0.88)

    roi = image[
        y_start:y_end,
        :
    ]

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    # -------------------------------------------------------------
    # Threshold giống lane follower
    # -------------------------------------------------------------

    mean_val = float(
        np.mean(gray)
    )

    if mean_val < 85.0:

        enhanced = self.clahe.apply(
            gray
        )

        _, thresh = cv2.threshold(
            enhanced,
            210,
            255,
            cv2.THRESH_BINARY
        )

    else:

        _, thresh = cv2.threshold(
            gray,
            195,
            255,
            cv2.THRESH_BINARY
        )

    thresh[:, :10] = 0
    thresh[:, -10:] = 0

    # -------------------------------------------------------------
    # Morphology nhẹ để nối các phần trắng rất gần nhau
    # -------------------------------------------------------------

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < 25:
            continue

        x, y, cw, ch = cv2.boundingRect(
            contour
        )

        if cw <= 0 or ch <= 0:
            continue

        aspect = (
            float(cw)
            / float(ch)
        )

        # ---------------------------------------------------------
        # Block ngang / chữ nhật
        # ---------------------------------------------------------

        if cw < 15:
            continue

        if ch < 5 or ch > 45:
            continue

        if aspect < 0.80:
            continue

        # Không lấy lane line quá dài
        if cw > int(w * 0.45):
            continue

        center_x = (
            x + cw / 2.0
        )

        center_y = (
            y + ch / 2.0
        )

        candidates.append(
            {
                "x": float(center_x),
                "y": float(center_y),
                "w": float(cw),
                "h": float(ch),
                "area": float(area)
            }
        )

    if len(candidates) < 3:
        return False, None

    # -------------------------------------------------------------
    # Gom các block cùng Y
    # -------------------------------------------------------------

    groups = []

    for candidate in candidates:

        added = False

        for group in groups:

            avg_y = float(
                np.mean(
                    [
                        p["y"]
                        for p in group
                    ]
                )
            )

            if abs(
                candidate["y"]
                - avg_y
            ) <= 18.0:

                group.append(
                    candidate
                )

                added = True

                break

        if not added:

            groups.append(
                [candidate]
            )

    best_group = None

    for group in groups:

        if len(group) < 3:
            continue

        min_x = min(
            p["x"] - p["w"] / 2.0
            for p in group
        )

        max_x = max(
            p["x"] + p["w"] / 2.0
            for p in group
        )

        span = (
            max_x - min_x
        )

        # ---------------------------------------------------------
        # Phải trải đủ rộng
        # ---------------------------------------------------------

        if span < w * 0.30:
            continue

        # ---------------------------------------------------------
        # Tâm nhóm không nằm sát mép
        # ---------------------------------------------------------

        group_center = (
            min_x + max_x
        ) / 2.0

        if group_center < w * 0.20:
            continue

        if group_center > w * 0.80:
            continue

        # ---------------------------------------------------------
        # Các block phải có phân bố ngang
        # ---------------------------------------------------------

        sorted_group = sorted(
            group,
            key=lambda p: p["x"]
        )

        valid_gap_count = 0

        for i in range(
            1,
            len(sorted_group)
        ):

            previous = sorted_group[i - 1]
            current = sorted_group[i]

            previous_right = (
                previous["x"]
                + previous["w"] / 2.0
            )

            current_left = (
                current["x"]
                - current["w"] / 2.0
            )

            gap = (
                current_left
                - previous_right
            )

            if gap >= -5.0:
                valid_gap_count += 1

        if valid_gap_count < 2:
            continue

        if (
            best_group is None
            or len(group)
            > len(best_group)
        ):

            best_group = group

    if best_group is None:
        return False, None

    line_y = float(
        np.mean(
            [
                p["y"]
                for p in best_group
            ]
        )
    )

    min_x = min(
        p["x"] - p["w"] / 2.0
        for p in best_group
    )

    max_x = max(
        p["x"] + p["w"] / 2.0
        for p in best_group
    )

    return True, {
        "y": line_y + y_start,
        "x_min": min_x,
        "x_max": max_x,
        "count": len(best_group)
    }


# =================================================================
# ĐIỀU KHIỂN QUA VẠCH
# =================================================================

def _white_dashed_cross_line_control_addon(
    self
):

    if self.image is None:
        return False

    now = time.monotonic()

    detected, info = (
        _detect_white_dashed_cross_line_addon(
            self,
            self.image
        )
    )

    h, w, _ = self.image.shape

    active = getattr(
        self,
        "white_cross_addon_active",
        False
    )

    last_seen = getattr(
        self,
        "white_cross_addon_last_seen",
        0.0
    )

    cooldown = getattr(
        self,
        "white_cross_addon_cooldown",
        0.0
    )

    # =============================================================
    # ĐANG QUA VẠCH
    # =============================================================

    if active:

        if detected:

            self.white_cross_addon_last_seen = now

            line_y = info["y"]

            # -----------------------------------------------------
            # Vạch còn xa
            # -----------------------------------------------------

            if line_y < h * 0.60:

                self.drive(
                    0.20,
                    0.0
                )

                return True

            # -----------------------------------------------------
            # Vạch gần
            # -----------------------------------------------------

            if line_y < h * 0.78:

                self.drive(
                    0.18,
                    0.0
                )

                return True

            # -----------------------------------------------------
            # Vạch rất gần
            # -----------------------------------------------------

            cross_start = getattr(
                self,
                "white_cross_addon_start",
                None
            )

            if cross_start is None:

                self.white_cross_addon_start = now

                cross_start = now

            elapsed = (
                now
                - cross_start
            )

            # Cho xe chạy thẳng lên vạch
            if elapsed < 0.70:

                self.drive(
                    0.14,
                    0.0
                )

                return True

            # -----------------------------------------------------
            # Đã qua
            # -----------------------------------------------------

            self.white_cross_addon_active = False

            self.white_cross_addon_start = None

            self.white_cross_addon_cooldown = (
                now + 1.5
            )

            self.get_logger().info(
                "WHITE DASHED CROSS LINE -> "
                "PASSED"
            )

            return False

        # =========================================================
        # MẤT VẠCH TẠM THỜI
        # =========================================================

        if (
            now - last_seen
            < 0.70
        ):

            self.drive(
                0.15,
                0.0
            )

            return True

        self.white_cross_addon_active = False
        self.white_cross_addon_start = None

        return False

    # =============================================================
    # CHƯA ACTIVE
    # =============================================================

    if now < cooldown:
        return False

    if not detected:
        return False

    line_y = info["y"]

    if (
        h * 0.35
        < line_y
        < h * 0.82
    ):

        self.white_cross_addon_active = True

        self.white_cross_addon_last_seen = now

        self.white_cross_addon_start = None

        self.get_logger().info(
            "WHITE DASHED CROSS LINE DETECTED -> "
            "DRIVE STRAIGHT"
        )

        self.drive(
            0.20,
            0.0
        )

        return True

    return False


# =================================================================
# CONTROL CUỐI
#
# Traffic -> White dashed line -> HW/lane control cũ
# =================================================================

_previous_control_before_white_cross = (
    PreciseLaneFollower.control
)


def _final_control_with_white_cross(
    self
):

    if self.image is None:
        return

    current_time = time.monotonic()

    # =============================================================
    # TRAFFIC
    # =============================================================

    detected = (
        self.detect_traffic_light_and_signs(
            self.image
        )
    )

    status = (
        self.update_object_confirmation(
            detected
        )
    )

    traffic_handled = (
        self.handle_traffic_control(
            status,
            current_time
        )
    )

    if traffic_handled:
        return

    # =============================================================
    # VẠCH TRẮNG SỌC
    # =============================================================

    if _white_dashed_cross_line_control_addon(
        self
    ):

        return

    # =============================================================
    # KHÔNG CÓ VẠCH
    # -> TRẢ QUYỀN CHO CONTROL HW + LANE CŨ
    # =============================================================

    _previous_control_before_white_cross(
        self
    )


PreciseLaneFollower.control = (
    _final_control_with_white_cross
)


# =================================================================
# MAIN
# =================================================================

def main(args=None):

    rclpy.init(
        args=args
    )

    node = PreciseLaneFollower()

    try:

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        pass

    finally:

        node.stop()

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':
    main()
