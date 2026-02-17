# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 22:12:22 2025

@author: Taallum
"""

#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node 
import time
import math
from sensor_msgs.msg import Image
import os
#from hsh_inference.msg import Modelstreet_linesput
import yaml
import cv2
import copy
from hshbot_py import visualize_utils as vu

from hsh_inference.msg import ModelOutput

num_lane_street_linesputs = 9 # 3*2 x-coords, 3 probabilities
num_box_street_linesputs  = 5 # 2 corners, 1 probability
num_start_stop_street_linesputs=6 # x1,x2,y1,y2 , 2 probabilities

class PointWithExistence:
    x = 0
    y = 0
    x_veh = 0
    y_veh = 0
    z_veh = 0
    exists = False

    def __repr__(self):
        return f"(x, y): ({self.x}, {self.y}), (x_veh, y_veh, z_veh): ({self.x_veh}, {self.y_veh}, {self.z_veh}), exists: {str(self.exists)}\n"

class LineWithProbability:
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0
    x1_veh = 0
    y1_veh = 0
    x2_veh = 0
    y2_veh = 0
    probability = 0.0

    def __repr__(self):
        return f"(x1, y1): ({self.x1}, {self.y1}), (x2, y2): ({self.x2}, {self.y2})\
            \n (x1_veh, y1_veh): ({self.x1_veh}, {self.y1_veh}), (x2_veh, y2_veh): ({self.x2_veh}, {self.y2_veh})\
            \n Line probability: {self.probability}\n"

    def getXatY(self, y):
        # in pixel coordinates:
        # x = (x2-x1)/(y2-y1)*(y-y1) + x1
        x = (self.x2-self.x1)/(self.y2-self.y1)*(y-self.y1) + self.x1
        return x

    def getYatX_veh(self, x_veh):
        # in vehicle coordinates:
        # x = (x2-x1)/(y2-y1)*(y-y1) + x1
        y_veh = (self.y2_veh-self.y1_veh)/(self.x2_veh-self.x1_veh)*(x_veh-self.x1_veh) + self.y1_veh # todo: add_veh everywhere
        return y_veh

class BoxWithProbability:
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0
    probability = 0.0
    def __repr__(self):
        return f"(x1, y1): ({self.x1}, {self.y1}), (x2, y2): ({self.x2}, {self.y2}), probability: {self.probability}"

class LineStartStopWithProbability:
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0
    probability_exist = 0.0
    probability_is_red = 0.0

class Modelstreet_linesPostProc:
    line_left   = LineWithProbability()
    line_center = LineWithProbability()
    line_right  = LineWithProbability()
    lane_left  = PointWithExistence()
    lane_right = PointWithExistence()
    orientation = 0.0

    person_box = BoxWithProbability()
    person_base = PointWithExistence()

    line_start_stop = LineStartStopWithProbability()
    line_start_base  = PointWithExistence()
    line_stop_base  = PointWithExistence()

    def __repr__(self):
        return f"line_left:\n{self.line_left}\nline_center:\n{self.line_center}\nline_right:\n {self.line_right}\
            \nlane_left:{self.lane_left}\nlane_right{self.lane_right}\nperson_box:{self.person_box}\nperson_base:{self.person_base}"

def sigmoid(z):
    y = 1.0 / (1.0 + math.exp(-z))
    return y

def post_process(street_linesput):
    #print("Debug: street_linesput:", type(street_linesput), street_linesput)  
    #print("Debug: num_lane_street_linesputs:", num_lane_street_linesputs)
    #print("Debug: num_box_street_linesputs:", num_box_street_linesputs)

    lane_street_linesput = street_linesput[:num_lane_street_linesputs]
    #print("Debug: lane_street_linesput:", type(lane_street_linesput), lane_street_linesput)

    box_street_linesput = street_linesput[num_lane_street_linesputs:num_lane_street_linesputs+num_box_street_linesputs]
    #print("Debug: box_street_linesput:", type(box_street_linesput), box_street_linesput)

    start_stop_street_linesput = street_linesput[num_lane_street_linesputs+num_box_street_linesputs:]
    #print("Debug: start_stop_street_linesput:", type(start_stop_street_linesput), start_stop_street_linesput)

    #if isinstance(start_stop_street_linesput, float):
    #   start_stop_street_linesput = [start_stop_street_linesput]

    #print("Debug : start_stop_street_linesput:", type(start_stop_street_linesput), start_stop_street_linesput)

    lane_street_linesput[2] = sigmoid(lane_street_linesput[2])
    lane_street_linesput[5] = sigmoid(lane_street_linesput[5])
    lane_street_linesput[8] = sigmoid(lane_street_linesput[8])

    box_street_linesput[4] = sigmoid(box_street_linesput[4])

    start_stop_street_linesput[4] = sigmoid(start_stop_street_linesput[4])
    start_stop_street_linesput[5] = sigmoid(start_stop_street_linesput[5])

    return lane_street_linesput, box_street_linesput, start_stop_street_linesput

'''def post_process(street_linesput):
    lane_street_linesput = street_linesput[:num_lane_street_linesputs]
    box_street_linesput  = street_linesput[num_lane_street_linesputs:num_lane_street_linesputs+num_box_street_linesputs]
    start_stop_street_linesput = street_linesput [num_lane_street_linesputs+num_box_street_linesputs]

    # apply sigmoid to classification street_linesputs
    lane_street_linesput[2] = sigmoid(lane_street_linesput[2])
    lane_street_linesput[5] = sigmoid(lane_street_linesput[5])
    lane_street_linesput[8] = sigmoid(lane_street_linesput[8])

    box_street_linesput[4]  = sigmoid(box_street_linesput[4])

    start_stop_street_linesput[4] =sigmoid(start_stop_street_linesput[4])
    start_stop_street_linesput[5] =sigmoid(start_stop_street_linesput[5])

    return lane_street_linesput, box_street_linesput,start_stop_street_linesput'''

def postproc_demo_2024(model_street_linesput_msg, calib = None, logger = None):
    if len(model_street_linesput_msg.data) != num_lane_street_linesputs + num_box_street_linesputs + num_start_stop_street_linesputs:
        if logger != None:
            logger.info("Error: Unexpected number of model outputs:")
            logger.info(f"- expected: {num_lane_street_linesputs} + {num_box_street_linesputs} + {num_start_stop_street_linesputs}")
            logger.info(f"- received: {len(model_street_linesput_msg.data)}")
        assert(False)
        
    lane_street_linesput, box_street_linesput, start_stop_street_linesput = post_process(model_street_linesput_msg.data)

    # fixed y-positions
    y1 = 150
    y2 = 200

    street_lines = Modelstreet_linesPostProc()
    street_lines.line_left.x1 = lane_street_linesput[0]
    street_lines.line_left.y1 = y1
    street_lines.line_left.x2 = lane_street_linesput[1]
    street_lines.line_left.y2 = y2
    street_lines.line_left.probability = lane_street_linesput[2]
    street_lines.line_center.x1 = lane_street_linesput[3]
    street_lines.line_center.y1 = y1
    street_lines.line_center.x2 = lane_street_linesput[4]
    street_lines.line_center.y2 = y2
    street_lines.line_center.probability = lane_street_linesput[5]
    street_lines.line_right.x1 = lane_street_linesput[6]
    street_lines.line_right.y1 = y1
    street_lines.line_right.x2 = lane_street_linesput[7]
    street_lines.line_right.y2 = y2
    street_lines.line_right.probability = lane_street_linesput[8]
    street_lines.lane_left.exists = False
    street_lines.lane_right.exists = False
    
    street_lines.person_box.x1 = box_street_linesput[0]
    street_lines.person_box.y1 = box_street_linesput[1]
    street_lines.person_box.x2 = box_street_linesput[2]
    street_lines.person_box.y2 = box_street_linesput[3]
    street_lines.person_box.probability = box_street_linesput[4]
    street_lines.person_base.x = (street_lines.person_box.x1 + street_lines.person_box.x2) / 2
    street_lines.person_base.y = max(street_lines.person_box.y1, street_lines.person_box.y2)
    street_lines.person_base.exists = street_lines.person_box.probability > 0.5

    street_lines.line_start_stop.x1 = start_stop_street_linesput[0]
    street_lines.line_start_stop.y1 = start_stop_street_linesput[1]
    street_lines.line_start_stop.x2 = start_stop_street_linesput[2]
    street_lines.line_start_stop.y2 = start_stop_street_linesput[3]
    
    street_lines.line_start_stop.probability_exist  = start_stop_street_linesput[4]
    street_lines.line_start_stop.probability_is_red = start_stop_street_linesput[5]

    street_lines.line_start_base.exists = False
    street_lines.line_stop_base.exists = False

    if (street_lines.line_start_stop.probability_exist > 0.5 and street_lines.line_start_stop.probability_is_red > 0.5):
        # start lane exists
        street_lines.line_start_base.x = (street_lines.line_start_stop.x1 + street_lines.line_start_stop.x2 ) / 2
        street_lines.line_start_base.y = (street_lines.line_start_stop.y1 + street_lines.line_start_stop.y2 ) / 2
        street_lines.line_start_base.exists = True

    if (street_lines.line_start_stop.probability_exist  > 0.5 and street_lines.line_start_stop.probability_is_red < 0.5):
        # stop lane exists
        street_lines.line_stop_base.x = (street_lines.line_start_stop.x1 + street_lines.line_start_stop.x2 ) / 2
        street_lines.line_stop_base.y = (street_lines.line_start_stop.y1 + street_lines.line_start_stop.y2 ) / 2
        street_lines.line_stop_base.exists = True

    assert(calib != None)
    xyz_veh = calib.img2veh(imagelinePointsToarray(street_lines))
    street_lines = vehCoordToLinePoints(xyz_veh, street_lines)

    PROB_THRESHHOLD = 0.6

    if ((street_lines.line_left.probability > PROB_THRESHHOLD) and (street_lines.line_center.probability > PROB_THRESHHOLD)):
        # left lane exists
        street_lines.lane_left.x_veh = (street_lines.line_left.x2_veh + street_lines.line_center.x2_veh ) / 2
        street_lines.lane_left.y_veh = (street_lines.line_left.y2_veh + street_lines.line_center.y2_veh ) / 2
        street_lines.lane_left.exists = True
        
    elif (street_lines.line_left.probability > PROB_THRESHHOLD) and not(street_lines.line_center.probability > PROB_THRESHHOLD):
        # left lane exists, but center line not valid
        street_lines.lane_left.x_veh = street_lines.line_left.x2_veh
        street_lines.lane_left.y_veh = street_lines.line_left.y2_veh  - 10
        street_lines.lane_left.exists = True
        
    elif (street_lines.line_center.probability > PROB_THRESHHOLD)and not(street_lines.line_left.probability > PROB_THRESHHOLD):    
        # left lane exists, but left line not valid
        street_lines.lane_left.x_veh = street_lines.line_center.x2_veh 
        street_lines.lane_left.y_veh = street_lines.line_center.y2_veh  + 10
        street_lines.lane_left.exists = True
   

    if ((street_lines.line_center.probability > PROB_THRESHHOLD) and (street_lines.line_right.probability > PROB_THRESHHOLD)):
        # right lane exists
        street_lines.lane_right.x_veh = (street_lines.line_right.x2_veh + street_lines.line_center.x2_veh ) / 2
        street_lines.lane_right.y_veh = (street_lines.line_right.y2_veh +  street_lines.line_center.y2_veh ) / 2   
        street_lines.lane_right.exists = True
    
    elif (street_lines.line_right.probability > PROB_THRESHHOLD)and not(street_lines.line_center.probability > PROB_THRESHHOLD):
        # right lane exists, but center line not valid
        street_lines.lane_right.x_veh = street_lines.line_right.x2_veh 
        street_lines.lane_right.y_veh = street_lines.line_right.y2_veh  + 10
        street_lines.lane_right.exists = True

    elif (street_lines.line_center.probability > PROB_THRESHHOLD)and not(street_lines.line_right.probability > PROB_THRESHHOLD):
         # right lane exists, but right line not valid
         street_lines.lane_right.x_veh = street_lines.line_center.x2_veh 
         street_lines.lane_right.y_veh = street_lines.line_center.y2_veh  - 10
         street_lines.lane_right.exists = True
    
    if (street_lines.lane_left.exists) or (street_lines.lane_right.exists):
        street_lines.orientation = np.arctan(
            (street_lines.line_center.y1_veh - street_lines.line_center.y2_veh) / 
            (street_lines.line_center.x1_veh - street_lines.line_center.x2_veh)
        )

    return street_lines

def imagelinePointsToarray(street_lines):
    uv_undist = np.array([
        [street_lines.line_left.x1, street_lines.line_left.x2, street_lines.line_center.x1, 
            street_lines.line_center.x2, street_lines.line_right.x1, street_lines.line_right.x2, street_lines.person_base.x,
            street_lines.line_start_base.x, street_lines.line_stop_base.x],
        [street_lines.line_left.y1, street_lines.line_left.y2, street_lines.line_center.y1, 
            street_lines.line_center.y2, street_lines.line_right.y1, street_lines.line_right.y2, street_lines.person_base.y,
            street_lines.line_start_base.y, street_lines.line_stop_base.y]
    ])
    return uv_undist

def vehLinePointsToarray(street_lines):
    xyz_veh = np.array([
        [street_lines.line_left.x1_veh, street_lines.line_left.x2_veh, street_lines.line_center.x1_veh, 
            street_lines.line_center.x2_veh, street_lines.line_right.x1_veh, street_lines.line_right.x2_veh],
        [street_lines.line_left.y1_veh, street_lines.line_left.y2_veh, street_lines.line_center.y1_veh, 
            street_lines.line_center.y2_veh, street_lines.line_right.y1_veh, street_lines.line_right.y2_veh]
    ])
    return xyz_veh

def vehCoordToLinePoints(xyz_veh, street_lines ):
    detection_points_veh =  copy.deepcopy(street_lines)
    detection_points_veh.line_left.x1_veh = xyz_veh[0,0]
    detection_points_veh.line_left.y1_veh = xyz_veh[1,0]
    detection_points_veh.line_left.x2_veh= xyz_veh[0,1]
    detection_points_veh.line_left.y2_veh = xyz_veh[1,1]
    
    detection_points_veh.line_center.x1_veh = xyz_veh[0,2]
    detection_points_veh.line_center.y1_veh = xyz_veh[1,2]
    detection_points_veh.line_center.x2_veh = xyz_veh[0,3]
    detection_points_veh.line_center.y2_veh = xyz_veh[1,3]
    
    detection_points_veh.line_right.x1_veh = xyz_veh[0,4]
    detection_points_veh.line_right.y1_veh = xyz_veh[1,4]
    detection_points_veh.line_right.x2_veh = xyz_veh[0,5]
    detection_points_veh.line_right.y2_veh = xyz_veh[1,5]
       
    if (detection_points_veh.person_base.exists):
        detection_points_veh.person_base.x_veh = xyz_veh[0,6]
        detection_points_veh.person_base.y_veh = xyz_veh[1,6]
    
    if (detection_points_veh.line_start_base.exists):
        detection_points_veh.line_start_base.x_veh = xyz_veh[0,7]
        detection_points_veh.line_start_base.y_veh = xyz_veh[1,7]

    if (detection_points_veh.line_stop_base.exists):
        detection_points_veh.line_stop_base.x_veh = xyz_veh[0,8]
        detection_points_veh.line_stop_base.y_veh = xyz_veh[1,8]

    return detection_points_veh

class PerspectiveCalibration:
    """
        Coordinate System Visualization:

                - Camera Coordinate System:

                                  O---------- Z_cam (camera forward)
                                / | 
                               /  |
                              /   |
                             /    | Y_cam (downwards)
                            /    
               X_cam (right)    

        - Vehicle Coordinate System:
            
                    Z_veh (upwards)
                                   |   Y_veh (left)
                                   |  /       
                                   | /
                                   |/                        
                                   O---------- X_veh (forward)
                  
        Note:
            - The camera is tilted 11° downward towards the ground.
            - The camera is mounted 13.5 cm above the ground.

            The transformation matrix aligns these two frames by rotating and translating the camera's perspective to match the vehicle's frame.
        """
    
    def __init__(self):
        # Set the directory for the configuration file and retrieve the YAML path
        savedir ="/home/jetson/workspace/videodrive_ws/src/hsh_camera_jetbot/config/"
        yaml_path =  savedir + os.getenv("ROS_CAM_CALIB", default = 1)
        
        # Load camera calibration data from YAML file
        with open(yaml_path, 'r') as f:
            calib_dict = yaml.load(f, Loader=yaml.SafeLoader)
        
        self.calib_width = calib_dict["camera"]["image_width"]
        self.calib_height = calib_dict["camera"]["image_height"]
        
        # Load intrinsic camera parameters (camera matrix and distortion coefficients)
        print("Intrinsic parameters:")
        self.cam_mtx = np.array(calib_dict["camera"]["intrinsics"]["camera_matrix"])
        print("Camera Matrix:\n", self.cam_mtx)
        #self.dist = np.array(calib_dict["camera"]["intrinsics"]["distortion"])
        #print("Distortion Coefficients:\n", self.dist)
        self.virtual_camera_zoom = calib_dict["camera"]["intrinsics"]["virtual_camera_zoom"]
        print("Virtual camera zoom:\n", self.virtual_camera_zoom)
        
        # apply virtual camera zoom
        self.cam_mtx[0,0] /= self.virtual_camera_zoom
        self.cam_mtx[1,1] /= self.virtual_camera_zoom
        
        # Compute the inverse of the camera matrix for later transformations
        self.inverse_cam_mtx = np.linalg.inv(self.cam_mtx)
        
        # Extrinsic parameter: compute the transformation matrix from camera to vehicle frame
        print("\n\nExtrinsic Parameter:")
        self.h_cam = np.array(calib_dict["camera"]["extrinsics"]["distance_to_ground_m"])
        self.theta_cam = np.array(calib_dict["camera"]["extrinsics"]["angle_to_ground_deg"])

        # Set up camera-to-vehicle rotation (Roll-Pitch-Yaw transformation) using the specified camera angle
        # First rotate around Z-axis -90° then around X-axis -90°-theta
        print("Theta: ", self.theta_cam)
        R_cam2veh = self.RPY(np.array([(-90 - self.theta_cam) * math.pi / 180.0, 0, -90 * math.pi / 180.0]))
        
        # Translation vector, with camera mounted 13.5 cm above the ground
        t_cam2veh = np.array([0, 0, self.h_cam * 100]) # in cm
        
        # Construct the camera-to-vehicle transformation matrix
        self.T_cam2veh = np.vstack((np.hstack((R_cam2veh, t_cam2veh[:, None])), [0, 0, 0, 1]))
        print("Camera to Vehicle Transformation:\n", self.T_cam2veh)
        
        # Compute the inverse transformation matrix from vehicle to camera frame
        R, t = self.T_cam2veh[:3, :3], self.T_cam2veh[:3, 3]
        T_inv = np.eye(4)
        T_inv[:3, :3] = R.T
        T_inv[:3, 3] = -R.T @ t
        self.T_veh2cam  = T_inv
        print("\nVehicle to Camera Transformation:\n", self.T_veh2cam)
        
        # Camera position and normal vector in the vehicle frame
        self.x0_cam = self.T_veh2cam.dot(np.array([0., 0., 0., 1]))  # Origin of vehicle coordinate system in Camera frame
        self.n_cam  = self.T_veh2cam[:3,:3].dot(np.array([0., 0., 1]))

        # Test horizon (should be v ~= 82)
        xyz_veh_horizon = np.array(((1000., 0., 0.), )).T  # 1000cm in front of the camera
        uv_horizon = self.veh2img(xyz_veh_horizon)
        print("u_horizon = ", uv_horizon[0])
        print("v_horizon = ", uv_horizon[1])
        
    def RPY(self, theta):
        """Compute rotation matrix from Roll-Pitch-Yaw angles."""
        # Rotation around X-axis
        R_x = np.array([[1, 0, 0],
                        [0, math.cos(theta[0]), -math.sin(theta[0])],
                        [0, math.sin(theta[0]), math.cos(theta[0])]])
        
        # Rotation around Y-axis
        R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                        [0, 1, 0],
                        [-math.sin(theta[1]), 0, math.cos(theta[1])]])
        
        # Rotation around Z-axis
        R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                        [math.sin(theta[2]), math.cos(theta[2]), 0],
                        [0, 0, 1]])
        
        # Combined rotation matrix
        R = np.dot(R_z, np.dot(R_y, R_x))
        return R
            

    def img2veh(self, detection_points):
        """
        Transform image points to vehicle coordinates.
        """
        detection_points[0, :] = detection_points[0, :] / 224 * self.calib_width
        detection_points[1, :] = detection_points[1, :] / 224 * self.calib_height
        
        # Transform points from pixel to normalized camera coordinates
        xy1_cam = self.inverse_cam_mtx.dot(np.insert(detection_points, 2, 1, axis=0))
        
        # Add a homogeneous coordinate to the points (X/Z, Y/Z, 1, 1)
        xy1_cam = np.insert(xy1_cam, 3, 1, axis=0)
        
        # intersection with calibration surface:
        ncam_norm = self.n_cam 
        denominator = xy1_cam[:3, :].T.dot(ncam_norm.reshape(3, 1))
        numerator = self.x0_cam[:3].T.dot(ncam_norm.reshape(3, 1))
        z_cam = numerator / denominator
        
        # Convert to 3D coordinates in the camera frame
        xyz_cam_din = z_cam * xy1_cam[:3, :].T
        xyz_cam_din = np.insert(xyz_cam_din, 3, 1, axis=1)
        
        # Transform the points from camera to vehicle coordinates
        xyz_veh = self.T_cam2veh.dot(xyz_cam_din.T)
        return xyz_veh[:3,:]
    
    def veh2img(self, xyz_veh):
        xyz_veh = np.insert(xyz_veh, 3, 1, axis=0)
        xyz_cam_din = self.T_veh2cam.dot(xyz_veh)
        xy1_cam =  xyz_cam_din[:3, :] / xyz_cam_din[2, :]
        uv1_undist =  self.cam_mtx.dot(xy1_cam) 
        uv_undist = uv1_undist[:2,:]
        uv_undist[0,:] = uv_undist[0,:] * 224 / self.calib_width
        uv_undist[1,:] = uv_undist[1,:] * 224 / self.calib_height
        
        return uv_undist[:2 , : ]


# ======================================================================
#   NEW: YOLO CAR DETECTION POST-PROCESSING (Format A, 640x640 -> 224x224)
# ======================================================================

class CarBase:
    """
    Holds information about a single detected car.
    Coordinates x_min, y_min, x_max, y_max are in the 224x224 model space.
    """
    def __init__(self):
        self.x_min = 0.0
        self.y_min = 0.0
        self.x_max = 0.0
        self.y_max = 0.0
        self.confidence = 0.0
        self.class_id = -1
        self.exists = False
        self.x_veh = 0.0   # forward distance [m]
        self.y_veh = 0.0   # lateral distance [m]
    def __repr__(self):
        return (
            f"CarBase(x_min={self.x_min:.1f}, y_min={self.y_min:.1f}, "
            f"x_max={self.x_max:.1f}, y_max={self.y_max:.1f}, "
            f"conf={self.confidence:.2f}, cls={self.class_id}, exists={self.exists})"
        )

class ModelCarPostProc:
    """
    Container for car detection output, similar style to Modelstreet_linesPostProc.
    """
    def __init__(self):
        self.car_base = CarBase()

    def __repr__(self):
        return f"ModelCarPostProc(car_base={self.car_base})"

def postproc_car_detection(model_msg, calib=None, logger=None, conf_thresh=0.5):
    out = ModelCarPostProc()

    if len(model_msg.dim_out) != 3:
        return out

    batch, num_det, stride = model_msg.dim_out
    expected_len = batch * num_det * stride

    if stride != 6 or len(model_msg.data) != expected_len:
        return out

    data = np.array(model_msg.data, dtype=np.float32).reshape(num_det, stride)

    # choose best detection
    best_idx = -1
    best_conf = -1.0

    for i in range(num_det):
        x1, y1, x2, y2, conf, cls_id = data[i]
        if conf > conf_thresh and conf > best_conf:
            best_idx = i
            best_conf = float(conf)

    if best_idx < 0:
        return out

    x1, y1, x2, y2, conf, cls_id = data[best_idx]

    # scale 640 → 224
    scale = 224.0 / 640.0
    x1_224 = x1 * scale
    y1_224 = y1 * scale
    x2_224 = x2 * scale
    y2_224 = y2 * scale

    out.car_base.x_min = x1_224
    out.car_base.y_min = y1_224
    out.car_base.x_max = x2_224
    out.car_base.y_max = y2_224
    out.car_base.confidence = conf
    out.car_base.class_id = int(cls_id)
    out.car_base.exists = True

    # ================================
    # Convert to vehicle coordinates
    # ================================
    if calib is not None:
        u = (x1_224 + x2_224) / 2.0  # center horizontally
        v = y2_224                   # bottom of bbox

        pixel_point = np.array([[u], [v]])
        veh_coords = calib.img2veh(pixel_point)

        out.car_base.x_veh = float(veh_coords[0,0]/100)  # forward distance (meters)
        out.car_base.y_veh = float(veh_coords[1,0]/100)  # lateral position
    else:
        out.car_base.x_veh = 0.0
        out.car_base.y_veh = 0.0

    return out



if __name__ == '__main__':
    pass
