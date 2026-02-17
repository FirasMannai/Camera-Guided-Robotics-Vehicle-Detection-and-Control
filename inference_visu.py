#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node 
import time
import math
from sensor_msgs.msg import Image
from hshbot_py import ros_img_conversion as ric
from hsh_motors.msg import RobotTargetSpeed
from hsh_imgproc.msg import LaneMarkerArray
from std_msgs.msg import String, Int32MultiArray, Float32MultiArray, MultiArrayLayout, MultiArrayDimension
from std_msgs.msg import Bool as Bool_ros
from sensor_msgs.msg import PointCloud2

import cv2
import message_filters
from hsh_inference.msg import ModelOutput
import array
import pdb
from hshbot_py import inference_postproc as ip
from hshbot_py import visualize_utils as vu

# Required other nodes:
# ros2 run hsh_motors hsh_motors                   # start motor node
# ros2 launch hsh_camera_jetbot camera.launch.yaml # start camera node
# ros2 run rqt_image_view rqt_iresizemage_view           # start image viewer

class LaneAndObjDetection(Node):
    def __init__(self):
        super().__init__('lane_and_obj_detection')
        #create a publisher
        #self.publisher_ = self.create_publisher(String,'jetbot_motors/cmd_str',1)
        #self.publisher_ = self.create_publisher(RobotTargetSpeed, '/hsh_motors/setSpeed', 1)
        self.lane_detection_pub = self.create_publisher(ModelOutput, '/lane_following/center_points_x', 1)
        self.visu_img_pub = self.create_publisher(Image, '/inference/visu_img', 1)
       # self.pedestrian_detection_pub = self.create_publisher(String, '/hsh_collision_detection', 1)
        #self.MaxSpeedTrans = 350 # 350 mm/s are circa 700 counts per second at the decoder
        #self.SetSpeedTrans = 0.5 * self.MaxSpeedTrans
        
        #create a subscriber
        self.image_subscription = message_filters.Subscriber(self, Image, '/hsh_camera/img') 
        self.out_model_subscription = message_filters.Subscriber(self, ModelOutput, '/hsh_inference/yolo')
        ts = message_filters.TimeSynchronizer([self.image_subscription, self.out_model_subscription], 10)
        ts.registerCallback(self.yolo_callback)

        self.lane_target = None
        self.lane_target_sub = self.create_subscription(Int32MultiArray, '/control/lane_target_point', self.laneTarget_cb, 1)

        self.person_detected = False
        self.person_detected_sub = self.create_subscription(Bool_ros, '/control/person_detected', self.personDetected_cb, 1)

        self.lane_start_stop_detected = False
        self.lane_start_stop_detected_sub = self.create_subscription(Bool_ros, '/control/lane_start_stop_detected', self.lane_start_stop_detected_cb, 1)


        self.calib_obj = ip.PerspectiveCalibration()
        self.visu_bev_pub = self.create_publisher(Image, '/inference/visu_bev_img', 1)
        #self.veh_coord_pub = self.create_publisher(PointCloud2, '/inference/veh_coordinates', 1)

        self.get_logger().info("Init done")
        self.t1=0        

    def image_callback(self, image_msg, model_output_msg):
        # if int(msg.header.frame_id) % 2 != 0:
        #     #self.get_logger().info("Skipping image.")
        #     return
        #print("Bild wird verarbeitet")
        t1 = time.time()
        image = ric.image_to_numpy(image_msg)
        model_out = ip.postproc_demo_2024(model_output_msg, self.calib_obj)

        # visu data
        FONTSCALE = 1.5e-3
        THICKNESS_SCALE = 1e-3
        height = image.shape[0]
        width = image.shape[1]
        font_scale = min(width, height) * FONTSCALE
        thickness = 1 # math.ceil(min(width, height) * THICKNESS_SCALE)

        # draw detected lines
        i = 0
        for line in (model_out.line_left, model_out.line_center, model_out.line_right):
            if line.probability > 0.5:
                image = cv2.circle(image, (int(line.x1), line.y1), 6, (0, 200, 0), -1)
                image = cv2.circle(image, (int(line.x2), line.y2), 6, (0, 200, 0), -1)
                image = cv2.line(image,(int(line.x1), line.y1), (int(line.x2), line.y2), (0, 200, 0), 2)
                cv2.putText(image, "p_line"+ f"{i}" + " = " +  f"{line.probability*100:.1f}%", (0, 10 + i*10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0 ,200, 0), thickness)
            i += 1
       
        # draw lane target
        if self.lane_target != None:
                image = cv2.circle(image, (self.lane_target[0], self.lane_target[1]), 6, (0, 200, 200), -1)
                image = cv2.line(image,(112,224), (self.lane_target[0], self.lane_target[1]), (0, 200, 200), 2)
            
        #for lane in (model_out.lane_left, model_out.lane_right):
        #    if lane.exists:
        #        image = cv2.circle(image, (int(lane.x), int(lane.y)), 6, (0, 200, 200), -1)
        #        image = cv2.line(image,(112,224), (int(lane.x), int(lane.y)), (0, 200, 200), 2)
          
         # draw box
        p_obj = model_out.person_box.probability
        if model_out.person_box.probability > 0.5:
            if self.person_detected:
                box_color = (200, 0, 0)
            else:
                box_color = (0, 200, 0)
            image = cv2.rectangle(image, (int(model_out.person_box.x1), int(model_out.person_box.y1)), (int(model_out.person_box.x2), int(model_out.person_box.y2)), box_color, thickness=1)
            cv2.putText(image, "p_obj  = " + f"{p_obj *100:.1f}%", (0, 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), thickness)
        
        #draw start line
        p_is_red = model_out.line_start_stop.probability_is_red
        p_exist = model_out.line_start_stop.probability_exist
        if (model_out.line_start_stop.probability_exist > 0.5 and model_out.line_start_stop.probability_is_red>0.5):
            # if self.lane_start_stop_detected:
            #     line_color = (200, 0, 0)
            # else:
            #     line_color = (0, 200, 0)
            image = cv2.line(image, (int(model_out.line_start_stop.x1), int(model_out.line_start_stop.y1)), (int(model_out.line_start_stop.x2), int(model_out.line_start_stop.y2)), (0,0,200), thickness=1)
            cv2.putText(image, "p_startline  = " + f"{p_is_red * p_exist * 100:.1f}%", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), thickness)

    

    #    draw stop line

        if (model_out.line_start_stop.probability_model_output_msgexist >0.5 and model_out.line_start_stop.probability_is_red < 0.5):
            # if self.lane_start_stop_detected:
            #     line_color = (200, 0, 0)
            # else:
            #     line_color = (0, 200, 0)
            image = cv2.line(image, (int(model_out.line_start_stop.x1), int(model_out.line_start_stop.y1)), (int(model_out.line_start_stop.x2), int(model_out.line_start_stop.y2)), (200,0,0), thickness=1)
            cv2.putText(image, "p_stopline  = " + f"{p_exist*(1-p_is_red) *100:.1f}%", (0, 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), thickness)

        # image = cv2.line(image, (int(model_out.line_start_stop.x1), int(model_out.line_start_stop.y1)), (int(model_out.line_start_stop.x2), int(model_out.line_start_stop.y2)), (200,0,0), thickness=1)


        if self.visu_img_pub.get_subscription_count() > 0:
            # publish image with detection 
            img_out = Image()
            #t2 = time.time()
            img_out = ric.numpy_to_image(image, "rgb8")
            img_out.header.stamp = image_msg.header.stamp
            img_out.header.frame_id = image_msg.header.frame_id
            self.visu_img_pub.publish(img_out)
       
        if self.visu_bev_pub.get_subscription_count() > 0:
            xyz_veh = ip.vehLinePointsToarray(model_out)
            bev_img = vu.visualize_points_with_arrows(xyz_veh)
            # publish image with detection 
            bev_msg = Image()
            t2 = time.time()
            bev_msg = ric.numpy_to_image(bev_img, "rgb8")
           # print("Laufzeit numpy_to_image2 ",time.time()-t2)
            bev_msg.header.stamp = image_msg.header.stamp
            bev_msg.header.frame_id = image_msg.header.frame_id
            self.visu_bev_pub.publish(bev_msg)
            print("Laufzeit numpy_to_image2 ",time.time()-t2)
            
            # Use ros_numpy to convert the numpy array into a PointCloud2 message
            #point_cloud_msg = ric.create_point_cloud_msg(xyz_veh[:3,:].T, image_msg.header)
            #self.veh_coord_pub.publish(point_cloud_msg)
        print("Laufzeit",time.time()-t1)

    def sigmoid(self, z):
        y = 1.0 / (1.0 + math.exp(-z))
        return y
    
    def yolo_callback(self, image_msg, yolo_msg):
        
        print("Laufzeit",time.time()-self.t1)
        
        # ---------------------------------------
        # 1. Convert ROS image → numpy
        # ---------------------------------------
        img = ric.image_to_numpy(image_msg)
        H, W = img.shape[:2]

        # Keep original image for drawing
        #orig_img = img.copy()

        # YOLO model input is 640 × 640
        INPUT_WIDTH = np.array(yolo_msg.dim_in)[2]
        INPUT_HEIGHT = np.array(yolo_msg.dim_in)[3]
        MODEL_OUT_SIZE = np.array(yolo_msg.dim_out)[2]
        MODEL_OUT_CLASSES = np.array(yolo_msg.dim_out)[1]


        
        #img_resized = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
        

        scale_x = W / INPUT_WIDTH
        scale_y = H / INPUT_HEIGHT

        # ---------------------------------------
        # 2. Read TensorRT output
        #    expected: 300 × 6
        # ---------------------------------------
        data = np.array(yolo_msg.data, dtype=np.float32)


        if data.size % MODEL_OUT_SIZE != 0:
            rclpy.logging.get_logger("yolo_callback").error(
                f"Invalid YOLO output size: {data.size}"
            )
            return img
        
       
        out = data.reshape(-1, MODEL_OUT_SIZE)    # (300, 6)

        CONF_THR = 0.50
        detections = []

        # ---------------------------------------
        # 3. Parse detections
        # ---------------------------------------
        for i in range(out.shape[0]):
            x1, y1, x2, y2, conf, cls_id = out[i]

            # skip empty rows
            if conf == 0:
                continue

            if conf < CONF_THR:
                continue

            # ---- Rescale from 640x640 → original size ----
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            # ---- Clip to image boundaries ----
            x1 = max(0, min(x1, W - 1))
            y1 = max(0, min(y1, H - 1))
            x2 = max(0, min(x2, W - 1))
            y2 = max(0, min(y2, H - 1))
            print(x1,y1,x2,y2)

            detections.append((x1, y1, x2, y2, conf, cls_id))

        print(f"Found {len(detections)} detections")

        # ---------------------------------------
        # 4. Draw bounding boxes
        # ---------------------------------------
        for (x1, y1, x2, y2, conf, cls_id) in detections:
            label = f"{int(cls_id)} {conf:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2),
                        (0, 255, 0), 2)

            cv2.putText(img, label,
                        (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

        # ---------------------------------------
        # 5. Publish visualization image
        # ---------------------------------------
        if self.visu_img_pub.get_subscription_count() > 0:
            
             
            msg_out = ric.numpy_to_image(img, "rgb8")
            
            msg_out.header.stamp = image_msg.header.stamp
            msg_out.header.frame_id = image_msg.header.frame_id
            self.visu_img_pub.publish(msg_out)
                   
        self.t1 = time.time()
        return img


    def laneTarget_cb(self, lane_target_msg):
        self.lane_target = lane_target_msg.data
        
    def personDetected_cb(self, person_detected_msg):
        self.person_detected = person_detected_msg.data

    def lane_start_stop_detected_cb(self, lane_start_stop_detected_msg):
        self.lane_start_stop_detected = lane_start_stop_detected_msg.data


def main(args = None):
    rclpy.init(args = args)
    node = LaneAndObjDetection()

    #create publisher to jetbot_motors to stop the motor, else the robot will
    #keep moving forward when the last state of the roboter was classified as 'free'
    #publisher = node.create_publisher(String,'jetbot_motors/cmd_str',1)
    #publisher = node.create_publisher(RobotTargetSpeed, '/hsh_motors/setSpeed', 1)
    
    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        #msg_out = String() 
        #msg_out.data = "stop"
        #publisher.publish(msg_out)

        print("Stopping motors...")
        msg_out = RobotTargetSpeed()
        msg_out.track_vel = 0.0
        msg_out.rot_vel   = 0.0
        msg_out.emergency_stop = True
        #publisher.publish(msg_out)
     
    node.destroy_node()
    rclpy.shutdown
    
if __name__ == '__main__':
    main()
