# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32MultiArray
from std_msgs.msg import Bool as Bool_ros

from readchar import readkey, key
from hsh_motors.msg import RobotTargetSpeed
from hsh_imgproc.msg import LaneMarkerArray
from hsh_inference.msg import ModelOutput
import threading
import math
from hshbot_py import new_inference_postproc as ip
from sensor_msgs.msg import PointCloud2
import numpy as np
import pdb
from hshbot_py import ros_img_conversion as ric

LANE_UNDEFINED = 0
LANE_LEFT  = 1
LANE_RIGHT = 2


class ControlNode(Node):
    """Reads the user input and communicates with the motor driver via the cmd_str topic
       and can be used as an easy way to take pictures with the onboard camera."""

    # class variables:
    speed_step_counter_rl = 0
    speed_step_counter_fb = 0
    steps = 4  # velocity steps in each direction (forward/backward/left/right)
    MaxSpeedTrans = 350  # 350 mm/s are circa 700 counts per second at the decoder
    MaxRotPerSec  = 0.7  # 0.7 U/s are circa 700 counts per second at the decoder
    DefaultSpeedTrans = 1.0 * MaxSpeedTrans
    SetSpeedTrans = DefaultSpeedTrans
    SetSpeedRot   = 0.0
    kp = 7.0

    mutex = threading.Lock()

    TargetLane = LANE_UNDEFINED
    OldLane = LANE_UNDEFINED
    target_lane_x = 0
    person_detected = False

    LaneChangeSituation = False
    LaneChangeAlpha = 0        # when lane change occurs: slide from LaneChangeOldXCenter to x_center_target
    LaneChangeOldXCenter = 0

    target_lane_y_veh = 0

    LaneChangeAlpha = 0
    LaneChangeOldYCenter_veh = 0

    # === Adaptive Cruise Control (ACC) parameters ===
    acc_active = False          # toggled by keyboard
    car_detected = False        # true if car ahead is detected
    car_distance = 0.0          # [m] forward distance to car (from postproc_car_detection)
    target_gap_distance = 0.5   # [m] desired distance
    acc_kp = 0.8                # proportional gain
    base_speed_trans = DefaultSpeedTrans  # [mm/s] base speed around which ACC works
    
    # Emergency stop parameters
    dist_tol = 0.05
    stop_dist = 0.30            # [m] emergency stop distance (30cm)
    resume_dist = 0.40          # [m] resume distance (40cm, hysteresis)

    def __init__(self):
        super().__init__('control_node')

        # shared calibration object for both lane+person and YOLO car detection
        self.calib_obj = ip.PerspectiveCalibration()

        # publisher to communicate with the motor driver
        self.publisher_ = self.create_publisher(RobotTargetSpeed, '/hsh_motors/setSpeed', 1)

        # create publisher for image writing
        self.img_publisher = self.create_publisher(String, '/camera_command', 1)

        # create publishers for inference_visu
        self.visu_lane_target_pub = self.create_publisher(Int32MultiArray, '/control/lane_target_point', 1)
        self.visu_lane_target_veh_pub = self.create_publisher(Int32MultiArray, '/control/lane_target_point_veh', 1)
        self.person_detected_pub = self.create_publisher(Bool_ros, '/control/person_detected', 1)
        self.start_stop_line_detected_pub = self.create_publisher(Bool_ros, '/control/start_stop_line_detected', 1)

        self.get_logger().info('Initialising keyboard control...')

        # === Lane + person hybrid model ===
        self.inference_sub = self.create_subscription(
            ModelOutput,
            '/hsh_inference/hybrid_lane_and_person',
            self.laneFollow_callback,
            1
        )

        # === ACC: subscribe to YOLO car detection model ===
        # Uses ip.postproc_car_detection(model_msg, calib=self.calib_obj, logger=...)
        self.car_detection_sub = self.create_subscription(
            ModelOutput,
            '/hsh_inference/yolo',
            self.car_detection_callback,
            1
        )

        # member variable for mode
        self.mode = "manual"
        self.get_logger().info('Reading from keyboard. Please ensure that the motor driver is running.')
        self.get_logger().info('---------------------')
        self.get_logger().info('Keys for saving images:')
        self.get_logger().info('- "1" : select lane 1 (left)')
        self.get_logger().info('- "2" : select lane 2 (right)')
        self.get_logger().info('- "3" : save image "blocked"')
        self.get_logger().info('- "4" : save image "free"')
        self.get_logger().info('- "5" : save image "other"')
        self.get_logger().info('- "c" : save (stereo) calib images')
        self.get_logger().info('Keys for robot control:')
        self.get_logger().info('- "arrow keys" : move robot')
        self.get_logger().info('- "a"     : automatic mode')
        self.get_logger().info('- "space" : manual mode / stop robot')
        self.get_logger().info('- "q"     : quit')
        self.get_logger().info('ACC control:')
        self.get_logger().info('- "g"     : toggle Adaptive Cruise Control ON/OFF')

        # Initialize ACC emergency state
        self.acc_emergency = False

        # key-handler thread
        key_thread = threading.Thread(target=self.handle_keys)
        key_thread.start()

    # === ACC: car detection callback ===
    def car_detection_callback(self, msg: ModelOutput):
        """
        Process car detection and compute distance in vehicle coordinates.

        Expects ip.postproc_car_detection(msg, calib) to return an object with:
          car_base.exists,
          car_base.x_min, car_base.x_max, car_base.y_min, car_base.y_max (in 224x224 space),
          car_base.x_veh, car_base.y_veh (in meters, forward / lateral).
        """
        if self.mode != "automatic":
            return

        car_out = ip.postproc_car_detection(msg, calib=self.calib_obj, logger=self.get_logger())

        if hasattr(car_out, 'car_base') and car_out.car_base.exists:
            self.car_detected = True
            self.car_distance = float(car_out.car_base.x_veh)  # meters

            self.get_logger().info(
                f"ACC: car detected at {self.car_distance:.2f} m ahead "
                f"(y_veh={car_out.car_base.y_veh:.2f} m)"
            )
        else:
            self.car_detected = False

    # === ACC: controller ===
    def adaptive_cruise_control(self):
        """ACC with integral control, emergency stop, and recovery."""
        if not (self.acc_active and self.mode == "automatic"):
            # Reset emergency state when ACC is disabled
            self.acc_emergency = False
            return

        # --- Safety distances (meters) ---
        target_dist = self.target_gap_distance    # desired distance (50 cm)
        dist_tol = self.dist_tol     # deadband ±5 cm
        stop_dist = self.stop_dist     # emergency stop at 30 cm
        resume_dist = self.resume_dist   # resume at 40 cm (hysteresis)

        # ================= EMERGENCY STOP =================
        if self.car_detected and self.car_distance <= stop_dist:
            self.acc_emergency = True
            self.SetSpeedTrans = 0.0
            self.get_logger().warn(f"ACC EMERGENCY STOP at {self.car_distance:.2f}m")
            return

        # ================= RECOVERY =================
        if self.acc_emergency:
            if (not self.car_detected) or (self.car_distance < resume_dist):
                self.SetSpeedTrans = 0.0
                return
            else:
                self.acc_emergency = False
                self.get_logger().info(f"ACC recovering at {self.car_distance:.2f}m")

        # ================= NORMAL ACC =================
        if not self.car_detected:
            # No car detected → use default lane-following speed
            self.SetSpeedTrans = float(self.DefaultSpeedTrans)
            return

        gap_error = float(self.car_distance - target_dist)

        if abs(gap_error) > dist_tol:
            delta = float(self.acc_kp * gap_error * 100.0)  # mm/s
            # Use explicit addition instead of += for clarity
            self.SetSpeedTrans = float(self.SetSpeedTrans + delta)
            
            # Optional logging
            self.get_logger().debug(
                f"[ACC] dist={self.car_distance:.2f}m, err={gap_error:.2f}m, "
                f"delta={delta:.1f}mm/s, speed={self.SetSpeedTrans:.0f}mm/s"
            )

        # --- Clamp speed (float-safe for ROS) ---
        self.SetSpeedTrans = float(
            max(80.0, min(self.SetSpeedTrans, self.MaxSpeedTrans))
        )

    # ------------------------------------------------------------------
    # Original functions (unchanged logic except where noted)
    # ------------------------------------------------------------------
    def setEgoLane(self, lane_left, lane_right):
        dist_left  = abs(lane_left.x - 112)
        dist_right = abs(lane_right.x - 112)
        if dist_left < dist_right:
            self.TargetLane = LANE_LEFT
            print("self.TargetLane = LANE_LEFT")
        else:
            self.TargetLane = LANE_RIGHT
            print("self.TargetLane = LANE_RIGHT")

    def setEgoLane_veh(self, lane_left, lane_right):
        dist_left  = abs(lane_left.y_veh)
        dist_right = abs(lane_right.y_veh)
        if dist_left < dist_right:
            self.TargetLane = LANE_LEFT
        else:
            self.TargetLane = LANE_RIGHT

    def handle_person_detection(self, model_out):
        if not model_out.person_base.exists:
            # if no person: default speed (ACC may overwrite later)
            self.SetSpeedTrans = float(self.DefaultSpeedTrans)
            return

        xp = model_out.person_base.x_veh
        yp = model_out.person_base.y_veh
        xp_min = 0

        person_detected_msg = Bool_ros()
        person_detected_msg.data = False

        if self.TargetLane == LANE_LEFT:
            if yp <= model_out.line_left.getYatX_veh(xp) and yp >= model_out.line_center.getYatX_veh(xp) and xp > xp_min:
                self.TargetLane = LANE_RIGHT
                self.LaneChangeSituation = True
                self.LaneChangeAlpha = 1
                self.LaneChangeOldYCenter_veh = self.target_lane_y_veh
                person_detected_msg.data = True

        elif self.TargetLane == LANE_RIGHT:
            if yp <= model_out.line_center.getYatX_veh(xp) and yp >= model_out.line_right.getYatX_veh(xp) and xp > xp_min:
                self.TargetLane = LANE_LEFT
                self.LaneChangeSituation = True
                self.LaneChangeAlpha = 1
                self.LaneChangeOldYCenter_veh = self.target_lane_y_veh
                person_detected_msg.data = True

        else:  # LANE_UNDEFINED - should not occur
            self.SetSpeedTrans = 0.0

        self.person_detected_pub.publish(person_detected_msg)

    def handel_start_stop_line_detection(self, model_out, msg_out: RobotTargetSpeed):
        if not model_out.line_start_base.exists:
            return

        stop_line_msg = Bool_ros()
        stop_line_msg.data = False

        if self.mode == "automatic" and model_out.line_start_base.x_veh <= 15:
            self.get_logger().info('start line detected')
            self.SetSpeedRot = 0.0
            self.SetSpeedTrans = 0.0
            self.mode = "manual"

            stop_line_msg.data = True
            self.start_stop_line_detected_pub.publish(stop_line_msg)

            msg_out.rot_vel  = float(self.SetSpeedRot)
            msg_out.track_vel = float(self.SetSpeedTrans)
            msg_out.emergency_stop = False
            self.publisher_.publish(msg_out)

            return

    def laneFollow_callback(self, msg: ModelOutput):
        # if self.mode != "automatic":
        #     return

        if self.mode == "manual":
            self.TargetLane = LANE_UNDEFINED

        msg_out = RobotTargetSpeed()

        model_out = ip.postproc_demo_2024(msg, self.calib_obj)
        self.handle_person_detection(model_out)

        # === ACC is applied after person handling, before steering ===
        if self.mode == "automatic":
            self.adaptive_cruise_control()

        if not model_out.lane_left.exists and not model_out.lane_right.exists:
            # no lane: no steering
            self.SetSpeedRot = 0.0

        else:  # at least one lane is detected
            if model_out.lane_left.exists and (not model_out.lane_right.exists):
                self.target_lane_y_veh = model_out.lane_left.y_veh

            elif model_out.lane_right.exists and (not model_out.lane_left.exists):
                self.target_lane_y_veh = model_out.lane_right.y_veh

            else:  # both lanes exist
                FRAME_RATE = 20  # fps
                time_for_lane_change = 2  # sec
                alpha_step = 1.0 / (time_for_lane_change * FRAME_RATE)

                if self.TargetLane == LANE_UNDEFINED:
                    self.setEgoLane_veh(model_out.lane_left, model_out.lane_right)

                if self.TargetLane == LANE_LEFT:
                    if self.LaneChangeSituation:
                        self.target_lane_y_veh = (
                            (1 - self.LaneChangeAlpha) * model_out.lane_left.y_veh +
                            self.LaneChangeAlpha * self.LaneChangeOldYCenter_veh
                        )
                        self.LaneChangeAlpha -= alpha_step
                        if self.LaneChangeAlpha < alpha_step and self.LaneChangeAlpha >= 0:
                            self.LaneChangeAlpha = 0
                        if self.LaneChangeAlpha < 0:
                            self.LaneChangeSituation = False
                            self.LaneChangeAlpha = 0
                    else:
                        self.target_lane_y_veh = model_out.lane_left.y_veh

                if self.TargetLane == LANE_RIGHT:
                    if self.LaneChangeSituation:
                        self.target_lane_y_veh = (
                            (1 - self.LaneChangeAlpha) * model_out.lane_right.y_veh +
                            self.LaneChangeAlpha * self.LaneChangeOldYCenter_veh
                        )
                        self.LaneChangeAlpha -= alpha_step
                        if self.LaneChangeAlpha < alpha_step and self.LaneChangeAlpha >= 0:
                            self.LaneChangeAlpha = 0
                        if self.LaneChangeAlpha < 0:
                            self.LaneChangeSituation = False
                            self.LaneChangeAlpha = 0
                    else:
                        self.target_lane_y_veh = model_out.lane_right.y_veh

                if self.mode == "automatic":
                    error = self.target_lane_y_veh / 100.0  # normalize with 1 m division

                    use_stainley_controller = True
                    use_pure_pursuit_controller = False

                    if use_stainley_controller:
                        heading_error = model_out.orientation
                        k_stainley = 3
                        wheel_base = 0.1485
                        speed = self.SetSpeedTrans / 1000.0  # mm/s -> m/s

                        steering = heading_error + np.arctan2(k_stainley * error, speed + 1e-5)
                        curvature = math.tan(steering) / wheel_base
                        v_rot = curvature * speed / (2 * math.pi)  # in rot/sec
                        self.SetSpeedRot = max(-self.MaxRotPerSec, min(v_rot, self.MaxRotPerSec))

                    elif use_pure_pursuit_controller:
                        heading_error = model_out.orientation
                        look_ahead_distance = 0.067
                        wheel_base = 0.1485
                        speed = self.SetSpeedTrans / 1000.0

                        if speed == 0:
                            speed += 1e-6
                        steering = heading_error + np.arctan2(
                            2 * wheel_base * error,
                            look_ahead_distance ** 2
                        )
                        curvature = math.tan(steering) / wheel_base
                        v_rot = curvature * speed / (2 * math.pi)
                        self.SetSpeedRot = max(-self.MaxRotPerSec, min(v_rot, self.MaxRotPerSec))

                    else:
                        rot_gain = self.MaxRotPerSec * self.kp
                        self.SetSpeedRot = rot_gain * error
                        self.SetSpeedRot = max(-self.MaxRotPerSec, min(self.SetSpeedRot, self.MaxRotPerSec))

        #self.handel_start_stop_line_detection(model_out, msg_out)

        if self.mode == "automatic":
            # set translational speed here, *after* ACC and lane-change scaling
            if self.LaneChangeSituation:
                msg_out.track_vel = float(self.SetSpeedTrans * 0.5)
            else:
                msg_out.track_vel = float(self.SetSpeedTrans)

            msg_out.rot_vel = float(self.SetSpeedRot)
            msg_out.emergency_stop = False
            self.publisher_.publish(msg_out)

        # --- Visualization of target lane point (vehicle coordinates) ---
        lane_target_veh_msg = Int32MultiArray()
        target_lane_x_veh = model_out.lane_left.x_veh  # (constant forward distance)
        lane_target_veh_msg.data = [int(target_lane_x_veh), int(self.target_lane_y_veh)]
        if self.visu_lane_target_veh_pub.get_subscription_count() > 0:
            self.visu_lane_target_veh_pub.publish(lane_target_veh_msg)

        # --- Visualization of target lane point (image coordinates) ---
        if self.visu_lane_target_pub.get_subscription_count() > 0:
            target_lane_points = np.array([[target_lane_x_veh], [self.target_lane_y_veh], [0]])
            uv_undist = self.calib_obj.veh2img(target_lane_points)
            lane_target_msg = Int32MultiArray()
            lane_target_msg.data = [int(uv_undist[0, 0]), int(uv_undist[1, 0])]
            self.visu_lane_target_pub.publish(lane_target_msg)

    def handle_keys(self):
        msg = RobotTargetSpeed()
        while True:
            try:
                k = readkey()  # Blocking wait for key
            except KeyboardInterrupt:
                self.get_logger().info("KeyboardInterrupt detected in key_thread. Exiting...")
                msg.track_vel = 0.0
                msg.rot_vel = 0.0
                with self.mutex:
                    self.publisher_.publish(msg)
                self.destroy_node()
                rclpy.shutdown()
                break

            if self.publisher_.get_subscription_count() == 0:
                self.get_logger().info('Warning: hsh_motors::VelocityCalculation not connected, yet.')

            # autonomous mode
            if k == "a":
                with self.mutex:
                    self.mode = "automatic"
                    self.get_logger().info('Mode: automatic')
                    self.TargetLane = LANE_UNDEFINED
                    self.LaneChangeSituation = False
                    self.LaneChangeAlpha = 0
                    self.LaneChangeOldYCenter_veh = 0
                    self.target_lane_y_veh = 0

            # ACC toggle
            elif k == "g":
                self.acc_active = not self.acc_active
                state = "ON" if self.acc_active else "OFF"
                self.get_logger().info(f"ACC {state}")

            # manual mode movement
            elif k == key.UP:
                if self.speed_step_counter_fb < self.steps:
                    self.speed_step_counter_fb += 1
                    self.get_logger().info('Forward one step faster')
                else:
                    self.get_logger().info('Can\'t drive Faster!')
                msg.track_vel = self.speed_step_counter_fb * (self.MaxSpeedTrans / self.steps)
                msg.rot_vel = self.speed_step_counter_rl * (self.MaxRotPerSec / self.steps)
                msg.emergency_stop = False
                with self.mutex:
                    self.publisher_.publish(msg)

            elif k == key.DOWN:
                if self.speed_step_counter_fb > -self.steps:
                    self.speed_step_counter_fb -= 1
                    self.get_logger().info('Backwards one step faster')
                else:
                    self.get_logger().info('Can\'t drive Faster!')
                msg.track_vel = self.speed_step_counter_fb * (self.MaxSpeedTrans / self.steps)
                msg.rot_vel = self.speed_step_counter_rl * (self.MaxRotPerSec / self.steps)
                msg.emergency_stop = False
                with self.mutex:
                    self.publisher_.publish(msg)

            elif k == key.LEFT:
                if self.speed_step_counter_rl < self.steps:
                    self.speed_step_counter_rl += 1
                    self.get_logger().info('Turning left one step faster')
                else:
                    self.get_logger().info('Can\'t rotate Faster!')
                msg.track_vel = self.speed_step_counter_fb * (self.MaxSpeedTrans / self.steps)
                msg.rot_vel = self.speed_step_counter_rl * (self.MaxRotPerSec / self.steps)
                msg.emergency_stop = False
                with self.mutex:
                    self.publisher_.publish(msg)

            elif k == key.RIGHT:
                if self.speed_step_counter_rl > -self.steps:
                    self.speed_step_counter_rl -= 1
                    self.get_logger().info('Turning right one step faster')
                else:
                    self.get_logger().info('Can\'t rotate Faster!')
                msg.track_vel = self.speed_step_counter_fb * (self.MaxSpeedTrans / self.steps)
                msg.rot_vel = self.speed_step_counter_rl * (self.MaxRotPerSec / self.steps)
                msg.emergency_stop = False
                with self.mutex:
                    self.publisher_.publish(msg)

            elif k == key.SPACE:
                with self.mutex:
                    self.mode = "manual"
                    self.get_logger().info('Mode: manual')

                self.speed_step_counter_rl = 0
                self.speed_step_counter_fb = 0
                msg.track_vel = 0.0
                msg.rot_vel = 0.0
                msg.emergency_stop = True

                with self.mutex:
                    self.get_logger().info('Stopped')
                    self.publisher_.publish(msg)

            elif k == "q":
                self.get_logger().info('Stopping robot...')
                msg.track_vel = 0.0
                msg.rot_vel = 0.0
                with self.mutex:
                    self.publisher_.publish(msg)
                self.destroy_node()
                rclpy.shutdown()
                break

            elif k == "0":
                self.TargetLane = LANE_UNDEFINED

            elif k == "1":
                if self.TargetLane != LANE_LEFT:
                    self.TargetLane = LANE_LEFT
                    self.LaneChangeSituation = True
                    self.LaneChangeAlpha = 1
                    self.LaneChangeOldYCenter_veh = self.target_lane_y_veh

            elif k == "2":
                if self.TargetLane != LANE_RIGHT:
                    self.TargetLane = LANE_RIGHT
                    self.LaneChangeSituation = True
                    self.LaneChangeAlpha = 1
                    self.LaneChangeOldYCenter_veh = self.target_lane_y_veh

            elif k == "3":
                msg_str = String()
                msg_str.data = "blocked"
                with self.mutex:
                    self.img_publisher.publish(msg_str)
                    self.get_logger().info('Take blocked image')

            elif k == "4":
                msg_str = String()
                msg_str.data = "free"
                with self.mutex:
                    self.img_publisher.publish(msg_str)
                    self.get_logger().info('Take free image')

            elif k == "5":
                msg_str = String()
                msg_str.data = "save"
                with self.mutex:
                    self.img_publisher.publish(msg_str)
                    self.get_logger().info('Take image')

            elif k == "c":
                msg_str = String()
                msg_str.data = "calib"
                with self.mutex:
                    self.img_publisher.publish(msg_str)
                    self.get_logger().info('Take calibration image')

            else:
                with self.mutex:
                    self.get_logger().info("Unknown key: '" + k + "'")


def main(args=None):
    try:
        rclpy.init(args=args)
        node = ControlNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()