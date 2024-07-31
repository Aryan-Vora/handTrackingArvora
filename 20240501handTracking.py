from cvzone.HandTrackingModule import HandDetector
import cv2
import mediapipe as mp
import time
import math
import socket
import numpy as np
from typing import List, Tuple
from xmlrpc.client import boolean
import serial
import struct
import threading
import numpy as np

class Tof:

    def __init__(self, port="COM3"): #後孔COM3
        # Initialize serial
        self.port = port
        self.serial = self._init_serial_port()
        # Initialize buffer
        self.buffer = bytearray()
        # Run/Stop flag
        self.run_flag = True

        # Start transmitting data
        self.start()
    

    def _init_serial_port(self) -> serial.Serial:
        """Initialize serial port"""
        ser = serial.Serial()
        ser.port = self.port
        # ser.baudrate = tof.static.BAUDRATE
        ser.timeout = 1
        try:
            ser.open()
        except:
            print(f"Failed to open port: {self.port}.")
            exit(1)
        return ser

    
    def start(self) -> None:
        """Start transmitting data, creates a parser thread."""
        self.send_command(command="settofpwr 0")
        self.send_command(command="settofpwr 1")
        self.send_command(command="settofconf 0")


    def stop(self) -> None:
        """Stop transmitting data"""
        self.send_command(command="settofpwr 0")

    
    def is_running(self) -> bool:
        return self.run_flag


    def send_command(self, command=None) -> None:
        """Send command to module, prompt if variable 'command' is left empty."""
        if command is None:
            command = input("Command to Module: ")
        print("Sending command: " + str(command))
        bytes_written = self.serial.write(bytes(command+chr(0x0d)+chr(0x0a),"ASCII"))
        print("Bytes written: " + str(bytes_written))

        self._get_response()
    

    # def _get_response(self) -> None:
    #     """Catch response after command is sent."""
    #     response = self.serial.read_until(terminator="\r\n")
    #     print(f"[RESPONSE] {response}")
    
    def _get_response(self) -> None:
        """Catch response after command is sent."""
        response = bytearray()
        while True:
            char = self.serial.read(1)
            if char == b'\r':
                continue
            elif char == b'\n':
                break
            response.extend(char)
        
        print(f"[RESPONSE] {response.decode('ASCII')}")


    def get_frame(self) ->  float:
        """Get a frame, tuple of lists of ndarray: (signal, range, status).
        Returns (None, None) if parsing failed.

        signal (peak rate per SPAD): ndarray of unsigned int (np.uintc)
        range (median range): ndarray of short (np.short)
        status (target status): ndarray of char (np.ubyte)"""

        # Read data
        self.buffer += self.serial.read(size=593)
        if len(self.buffer) < 593:
            return (None, None)
        # Cleanup previous tail
        ind = self.buffer.find(bytearray("DATA", encoding="ASCII"))
        self.buffer = self.buffer[ind:]
        # Align data
        self.buffer += self.serial.read(size=ind)
        if len(self.buffer) < 593:
            return (None, None)

        # Little Endian
        # 0-3: DATA
        # 4: Number of Zone
        # 5: Die Temperature
        # 6-9: Timestamp

        # 10-585: Signal data
        # 10 (+9*N): Number of Target
        # 11 - 14 (+9*N): peak_rate_kcps_per_spad(4 Bytes, little-endian)
        peak_rate_kcps_per_spad = np.zeros((8, 8), dtype=np.uintc)
        for i in range(64):
            peak_rate_kcps_per_spad[i//8][i%8] = struct.unpack("<I", self.buffer[9*i+11:9*i+15])[0]

        # 15 - 16 (+9*N): median_range
        median_range = np.zeros((8, 8), dtype=np.short)
        for i in range(64):
            median_range[i//8][i%8] = struct.unpack("<h", self.buffer[9*i+15:9*i+17])[0]

        for i in range(64): #除以40 變成公分
            median_range[i//8][i%8] = median_range[i//8][i%8] / 40.0
        
        ##Filter
        self.buffer = bytearray()
        #現實距離的10公分到50公分來做平均
        min_dist, max_dist = 10, 50 
        # Calculate the average depth within the specified range
        valid_depths = median_range[(median_range >= min_dist) & (median_range <= max_dist)]
        # if valid_depths.size > 0:
        #     average_depth = np.mean(valid_depths)
        #     return average_depth / 4  # Convert to millimeters
        # else:
        #     return 0
        
        # 多一個average_depth
        if valid_depths.size > 0:
            average_depth = np.mean(valid_depths)
            return median_range,average_depth,peak_rate_kcps_per_spad   # Convert to millimeters
        else:
            return median_range,0,peak_rate_kcps_per_spad
        
        
        # 17 (+9*N): Reflectance
        # 18 (+9*N): Target Status
        # target_status = np.zeros((8, 8), dtype=np.ubyte)
        # for i in range(64):
        #     target_status[i//8][i%8] = struct.unpack("<c", self.buffer[9*i+18:9*i+19])[0]
            
        # Clear buffer
        # self.buffer = bytearray()

        # return (peak_rate_kcps_per_spad, median_range)




def hand_angle(hand_):
    angle_list = []
    # 大拇指角度
    #thumb_angle = angle_2d_side(
    #    (hand_[0][0], hand_[0][1]),
    #    (hand_[2][0], hand_[2][1]),
    #    (hand_[4][0], hand_[4][1])
    #)
    #normalize_thumb_angle = normalize_angle(thumb_angle)
    ##angle_list.append(thumb_angle)
    #angle_list.append(normalize_thumb_angle)

    # 大拇指角度with向量
    thumb_vector1 = (hand_[0][0] - hand_[1][0], hand_[0][1] - hand_[1][1])
    thumb_vector2 = (hand_[3][0] - hand_[4][0], hand_[3][1] - hand_[4][1])
    thumb_angle = angle_2d_vectors(thumb_vector1, thumb_vector2)
    normalize_thumb_angle = normalize_angle(thumb_angle)
    angle_list.append(normalize_thumb_angle)
    #angle_list.append(thumb_angle)
    
    # 食指角度
    index_angle = angle_2d_side(
        (hand_[0][0], hand_[0][1]),
        (hand_[6][0], hand_[6][1]),
        (hand_[8][0], hand_[8][1])
    )
    normalize_index_angle = normalize_angle(index_angle)
    #angle_list.append(index_angle)
    angle_list.append(1 - normalize_index_angle)
    # 中指角度
    middle_angle = angle_2d_side(
        (hand_[0][0], hand_[0][1]),
        (hand_[10][0], hand_[10][1]),
        (hand_[12][0], hand_[12][1])
    )
    normalize_middle_angle = normalize_angle(middle_angle)
    #angle_list.append(middle_angle)
    angle_list.append(1 - normalize_middle_angle)
    # 無名指角度
    ring_angle = angle_2d_side(
        (hand_[0][0], hand_[0][1]),
        (hand_[14][0], hand_[14][1]),
        (hand_[16][0], hand_[16][1])
    )
    normalized_ring_angle = normalize_angle(ring_angle)
    #angle_list.append(ring_angle)
    angle_list.append(1 - normalized_ring_angle)
    # 小指角度
    pinky_angle = angle_2d_side(
        (hand_[0][0], hand_[0][1]),
        (hand_[18][0], hand_[18][1]),
        (hand_[20][0], hand_[20][1])
    )

    normalized_pinky_angle = normalize_angle(pinky_angle)
    #angle_list.append(pinky_angle)
    angle_list.append(1 - normalized_pinky_angle)
    return angle_list

def angle_2d_side(base, joint, tip):
    side_a = math.sqrt((joint[0] - base[0])**2 + (joint[1] - base[1])**2)
    side_b = math.sqrt((joint[0] - tip[0])**2 + (joint[1] - tip[1])**2)
    side_c = math.sqrt((tip[0] - base[0])**2 + (tip[1] - base[1])**2)
    return math.degrees(math.acos((side_a**2 + side_b**2 - side_c**2) / (2 * side_a * side_b)))

def angle_2d_vectors(vector1, vector2):
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)
    cosine_similarity = dot_product / (magnitude1 * magnitude2)
    return math.degrees(math.acos(cosine_similarity))

def normalize_angle(angle):
    return angle / 180.0

###計算手的旋轉角度


def calculate_rotation(hand_landmarks):
    wrist = np.array([hand_landmarks[0].x, hand_landmarks[0].y, hand_landmarks[0].z])
    index_bottom = np.array([hand_landmarks[5].x, hand_landmarks[5].y, hand_landmarks[5].z])
    middle_bottom = np.array([hand_landmarks[9].x, hand_landmarks[9].y, hand_landmarks[9].z])

    hand_vector = wrist - middle_bottom             # Mediapipe defines downwards as the positive direction for y
    hand_vector_y = index_bottom - middle_bottom    # The reference vector for rotation around y axis
    # hand_vector_normalized = hand_vector / np.linalg.norm(hand_vector)

    axes = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

    # Rotation on x axis is the angle between reference state (0, 1, 0) and the yz projection of the hand vector
    proj_x = (0, hand_vector[1], hand_vector[2])
    proj_x_normalized = proj_x / np.linalg.norm(proj_x)
    rotx = np.arccos(np.dot(axes[1], proj_x_normalized))
    rotx = np.arctan2(np.dot(np.cross(axes[1], proj_x), axes[0]), np.dot(axes[1], proj_x))

    
    # Rotation on y axis is the angle between reference state (1, 0, 0) and the xz projection of the hand vector
    # Hand vector is from index finger's bottom to middle finger's bottom
    proj_y = (hand_vector_y[0], 0, hand_vector_y[2])
    proj_y_normalized = proj_y / np.linalg.norm(proj_y)
    roty = np.arccos(np.dot(axes[0], proj_y_normalized))
    roty = np.arctan2(np.dot(np.cross(axes[0], proj_y), axes[1]), np.dot(axes[0], proj_y))
    
    # Rotation on z axis is the angle between reference state (0, 1, 0) and the xy projection of the hand vector
    
    proj_z = (hand_vector[0], hand_vector[1], 0)
    proj_z_normalized = proj_z / np.linalg.norm(proj_z)
    rotz = np.arccos(np.dot(axes[1], proj_z_normalized))
    rotz = np.arctan2(np.dot(np.cross(axes[1], proj_z), axes[2]), np.dot(axes[1], proj_z))

    rotation_x_deg = np.degrees(rotx)
    rotation_y_deg = np.degrees(roty)
    rotation_z_deg = np.degrees(rotz)

    rotation_z_deg = rotation_z_deg
    rotation_y_deg = rotation_y_deg
    

    return rotation_z_deg, rotation_y_deg, rotation_z_deg

############################Rotation z
#計算兩點距離來判斷z方向的rotation
def calculate_distance(hand_landmarks,i,j):
    # 提取關鍵點的座標
    wrist_x = hand_landmarks[i].x
    wrist_y = hand_landmarks[i].y
    middle_finger_x = hand_landmarks[j].x
    middle_finger_y = hand_landmarks[j].y
    
    # 計算距離
    distance = math.sqrt((middle_finger_x - wrist_x)**2 + (middle_finger_y - wrist_y)**2)
    
    return distance
    
#距離轉換成rotation z
def map_to_angle1(original_value):
    original_min = 28
    original_max = 8
    
    new_min = 90
    new_max = 0
    
    new_value = new_min + (original_value - original_min) * (new_max - new_min) / (original_max - original_min)
    
    return new_value

def calculate_angle_between_vectors(landmarks, point1, point2, point3, point4):
    # Get the coordinates of the four points
    x1, y1 = landmarks[point1][0], landmarks[point1][1]
    x2, y2 = landmarks[point2][0], landmarks[point2][1]
    x3, y3 = landmarks[point3][0], landmarks[point3][1]
    x4, y4 = landmarks[point4][0], landmarks[point4][1]

    # Calculate the vectors formed by (point1, point2) and (point3, point4)
    vector1 = (x2 - x1, y2 - y1)
    vector2 = (x4 - x3, y4 - y3)

    # Calculate the dot product of the two vectors
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]

    # Calculate the magnitudes of the two vectors
    magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

    # Calculate the cosine of the angle between the vectors
    cosine_theta = dot_product / (magnitude1 * magnitude2)

    # Calculate the angle in radians and then convert to degrees
    angle_in_radians = math.acos(cosine_theta)
    angle_in_degrees = math.degrees(angle_in_radians)

    return angle_in_degrees
##夾角轉換成rotation z
def map_to_angle2(original_value):
    original_min = 43
    original_max = 9.5
    
    new_min = 90
    new_max = 0
    
    new_value = new_min + (original_value - original_min) * (new_max - new_min) / (original_max - original_min)
    
    return new_value

def fixRotationz_1(rotation_z_distance, rotation_z_angle, rotation_x):
    if rotation_z_angle < 0:
        rotation_z_angle = 0
    if rotation_z_distance < 0:
        rotation_z_distance = 0
    # return min(rotation_z_distance, rotation_z_angle)
    if(abs(rotation_x) >= 39):
        return rotation_z_angle
    return rotation_z_distance



# def calculate_rotation_x(hand_landmarks, img_width, img_height):
#     # 計算手的中心座標
#     hand_center = (0, 0)
#     for landmark in hand_landmarks:
#         hand_center = (hand_center[0] + landmark.x * img_width, hand_center[1] + landmark.y * img_height)
#     hand_center = (hand_center[0] / len(hand_landmarks), hand_center[1] / len(hand_landmarks))
    
#     # 計算手的旋轉角度（相對於某一固定點，這裡假設固定點在圖像中心）
#     fixed_point = (img_width / 2, img_height / 2)
#     angle_rad = math.atan2(hand_center[1] - fixed_point[1], hand_center[0] - fixed_point[0])
#     angle_deg = math.degrees(angle_rad)
    
#     return -angle_deg

# def calculate_rotation_y(hand_landmarks):
#     # 計算手腕和食指指尖之间的向量
#     wrist = np.array([hand_landmarks[0].x, hand_landmarks[0].y, hand_landmarks[0].z])
#     index_tip = np.array([hand_landmarks[8].x, hand_landmarks[8].y, hand_landmarks[8].z])
    
#     hand_vector = index_tip - wrist

#     # 計算手的旋轉角度（繞Y轴的旋轉）
#     y_axis = np.array([0, 1, 0])  # Y軸的單位向量
#     hand_vector_normalized = hand_vector / np.linalg.norm(hand_vector)  #標準化手的向量
#     angle_rad_y = np.arccos(np.dot(y_axis, hand_vector_normalized))
#     angle_deg_y = np.degrees(angle_rad_y)

#     return angle_deg_y  

# def calculate_rotation_z(hand_landmarks):
#     # 計算手腕和中指指尖之間的向量
#     wrist = np.array([hand_landmarks[0].x, hand_landmarks[0].y, hand_landmarks[0].z])
#     middle_tip = np.array([hand_landmarks[12].x, hand_landmarks[12].y, hand_landmarks[12].z])
#     print(wrist, middle_tip)
#     hand_vector = middle_tip - wrist

#     #print(hand_landmarks[0].x, hand_landmarks[0].y, hand_landmarks[0].z)
    
#     # 計算手的旋轉角度（繞Z軸的旋轉）
#     z_axis = np.array([0, 0, 1])  # Z軸的單位向量
#     hand_vector_normalized = hand_vector / np.linalg.norm(hand_vector)  # 標準化手的向量
#     cross_product = np.cross(z_axis, hand_vector_normalized)  
#     angle_rad_z = np.arcsin(np.linalg.norm(cross_product))
#     angle_deg_z = np.degrees(angle_rad_z)

#     return angle_deg_z  # 返回角度值
#####

# depth calibration
# old method for 4 nearest neighbor mean

def handDetect(peak_rate, median_range):
    # peak_rate_kcps_per_spad
    result = []
    valid_median_range = 0  
    valid_count = 0  

    x = 0
    y = 0
    
    for i in range(8):
        row = []
        for j in range(8):
            # 篩選
            condition_met = peak_rate[i][j] >= 60000 and 10 <= median_range[i][j] <= 50
            row.append(condition_met)
            if condition_met:
                valid_median_range += median_range[i][j]
                valid_count += 1
                x += i
                y += j
        result.append(row)

    if valid_count > 0:
        average_valid_median_range = valid_median_range / valid_count
        x = x / valid_count
        y = y / valid_count
    else:
        average_valid_median_range = 0

    return result, average_valid_median_range,x,y
# 回傳手部區域的boolean二維陣列，以及有效的手部距離平均值

def landmarkDepth(hand_coords, median_range,handArea):
    landMarkdepth = []
    for lm in hand_coords:
        x = lm[0]
        y = lm[1]
        
        # determine which grid the point is in
        x = (x / 80.0) 
        y = (y / 80.0)
        
        # x y to fixed
        x0 = int(x)
        y0 = int(y)
    
        wx = x - x0
        wy = y - y0
        # 判斷x和y的小數部分，決定是向左或向右、向上或向下取整
        if wx >= 0.5:
            x1 = x0 + 1
        else :
            x1 = x0 -1
            
        if wy >= 0.5:
            y1 = y0 + 1
        else :
            y1 = y0 -1
    
        # 邊界檢查
        x0 = min(7, max(0, x0))
        x1 = min(7, max(0, x1))
        y0 = min(7, max(0, y0))
        y1 = min(7, max(0, y1))
        
        
        
        q11 = median_range[x0][y0]
        q12 = median_range[x1][y0]
        q21 = median_range[x0][y1]
        q22 = median_range[x1][y1]
    
        #現實距離的10公分到50公分來做平均
        # min_dist, max_dist = 400, 2000 
        
        # 已換成公分
        min_dist, max_dist = 10,50 
        
        # Calculate the average depth within the specified range
        valid_depths = []
        # if(q11<=max_dist and q11>=min_dist):
        #     valid_depths.append(q11)
        
        if(handArea[x0][y0]):
            valid_depths.append(q11)
        if(handArea[x1][y0]):
            valid_depths.append(q12)
        if(handArea[x0][y1]):
            valid_depths.append(q21)
        if(handArea[x1][y1]):
            valid_depths.append(q22)
        # if(q12<=max_dist and q12>=min_dist and abs(q11 - q12) < 5):
        #     valid_depths.append(q12)
        # if(q21<=max_dist and q21>=min_dist and abs(q11 - q21) < 5):
        #     valid_depths.append(q21)
        # if(q22<=max_dist and q22>=min_dist and abs(q11 - q22) < 5):
        #     valid_depths.append(q22)
        # 目前用4點平均
        
        #midean_range has already convert to cm
        if len(valid_depths) > 0:
                average_depth = np.mean(valid_depths)
                
                # landMarkdepth.append(average_depth / 4)  # Convert to millimeters
                
                landMarkdepth.append(average_depth ) # Convert to z in unity
                
                # return average_depth / 4  # Convert to millimeters
        else:
                landMarkdepth.append(0)
    
    return landMarkdepth

def calibration(x, y, fixed_x, fixed_y):
    cal_x = x * fixed_x[0] + fixed_x[1]
    cal_y = y * fixed_y[0] + fixed_y[1]
    cal_x = math.floor(cal_x)
    cal_y = math.floor(cal_y)
    if cal_x > 7:
        cal_x = 7
    if cal_x < 0:
        cal_x = 0
    if cal_y > 7:
        cal_y = 7
    if cal_y < 0:
        cal_y = 0
    return cal_x, cal_y

def world2Image(x, y, z, M, R, t,T):
    # 將圖像座標轉換為相機坐標系中的射線
    # imagePoint = np.array([x, y, z])
    imagePoint = np.array([x, y, z,1])
    
    
    if z == 0:
        image_coor = [0, 0, 0, 0]
    else: image_coor = imagePoint / z
    # image_coor = M.dot(image_coor)
    # image_coor = R.dot(image_coor)
    # print("image coor",image_coor)
    return M.dot(T.dot(image_coor))

def image2World(u, v, zConst,M,R,t):
    
    
    if u is None or v is None or zConst is None:
        return np.array([[0], [0], [0]])
    # 將圖像座標轉換為相機坐標系中的射線
    imagePoint = np.array([[u], [v], [1]])

    # 將射線轉換為相機坐標系中的點
    rightMatrix = np.linalg.inv(R).dot(np.linalg.inv(M).dot(imagePoint))

    # 計算深度對應的射線
    leftMatrix = np.linalg.inv(R).dot(t)
    s = (zConst + leftMatrix[2]) / rightMatrix[2]

    # 將深度應用於射線並轉換為世界座標系中的點
    cameraPoint = np.linalg.inv(R).dot(s * np.linalg.inv(M).dot(imagePoint) - t)

    return cameraPoint

def tof2World_fov(x, y, D):

    """
    計算物體在世界坐標系中的位置。

    參數:
    x, y: 物體在相機畫面中的位置（像素坐標）。
    W, H: 相機的解析度（寬度和高度）。
    FOV_h: 相機的水平視場角（度）。
    FOV_v: 相機的垂直視場角（度）。
    D: 物體到相機的距離。

    返回:
    (X, Y, Z): 物體在世界坐標系中的位置。
    """

    Cx = 3.5
    Cy = 3.5
    FOV_h = 45 
    FOV_v = 45

    # 將水平和垂直FOV從度轉換為弧度
    FOV_h_rad = math.radians(FOV_h/2)
    FOV_v_rad = math.radians(FOV_v/2)

    # print(FOV_h_rad)

    # 計算每像素的角度
    # angle_per_pixel_h = FOV_h_rad / W
    # angle_per_pixel_v = FOV_v_rad / H

    # 計算物體的水平和垂直角度
    theta_h = ((x - Cx) / Cx) * FOV_h_rad
    theta_v = ((y - Cy) / Cy) * FOV_v_rad

    # 計算物體在世界坐標系中的位置
    X = D * math.tan(theta_h)
    Y = D * math.tan(theta_v)
    Z = D

    return (X, Y, Z)
def add_nearby_points1(all_image_coords, tof_in_hand, img, imgWidth, imgHeight):
    # 計算 tof_in_hand 的第三個維度中位數深度
    tof_in_hand_depth_median = np.median([point[2] for point in tof_in_hand])
    
    # 計算中位數前後 7 公分的範圍
    depth_range = (tof_in_hand_depth_median - 7, tof_in_hand_depth_median + 7)
    
    
    # 選擇在範圍內的點
    tof_in_hand_filtered = [point for point in tof_in_hand if depth_range[0] <= point[2] <= depth_range[1]]
    tof_in_hand_np = np.array(tof_in_hand_filtered)
    
    # 再次計算中位數深度
    tof_in_hand_depth_median = np.median([point[2] for point in tof_in_hand_filtered])
    
    
    # 顯示中位數深度
    text = f"median depth: ({tof_in_hand_depth_median})"
    cv2.putText(img, text, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 遍歷所有影像坐標，找到符合條件的並添加到 tof_in_hand
    for image_coor in [coor for coor in all_image_coords if coor not in tof_in_hand_filtered]:
        x, y, z, i, j = image_coor
        
        # 檢查深度是否在範圍內
        if depth_range[0] <= z <= depth_range[1]:
            # 生成 9 宮格內的所有坐標
            neighbors = [(i+x_offset, j+y_offset) for x_offset in range(-1, 2) for y_offset in range(-1, 2)]
            
            # 檢查 9 宮格內是否有 tof_in_hand 的點
            for neighbor in neighbors:
                neighbor_i, neighbor_j = neighbor
                if 0 <= neighbor_i < 8 and 0 <= neighbor_j < 8:
                    neighbor_coords = [point for point in tof_in_hand_filtered if point[3] == neighbor_i and point[4] == neighbor_j]
                    if len(neighbor_coords) > 0 and depth_range[0] <= neighbor_coords[0][2] <= depth_range[1]:
                        tof_in_hand_filtered.append([x, y, z, i, j])
                        break
    
    return tof_in_hand_filtered

def get_depth_at_coord(x, y, tof_in_hand):
    if len(tof_in_hand) == 0:
        return None

    # Extract the x, y, and z coordinates from tof_in_hand
    tof_x = [point[0] for point in tof_in_hand]
    tof_y = [point[1] for point in tof_in_hand]
    tof_z = [point[2] for point in tof_in_hand]

    distances = []
    for i in range(len(tof_x)):
        distance = np.sqrt((tof_x[i] - x) ** 2 + (tof_y[i] - y) ** 2)
        distances.append((distance, tof_z[i]))

    distances.sort(key=lambda d: d[0])

    # Select the four closest points
    nearest_points = [distances[i][1] for i in range(min(4, len(distances)))]

    # Compute the average depth value of the four nearest points
    if len(nearest_points) > 0:
        mean_depth = np.mean(nearest_points)
        return mean_depth
    else:
        return 0

def get_depth_at_coord_old(x, y, tof_in_hand):
    if len(tof_in_hand) == 0:
        return None

    # Extract the x, y, and z coordinates from tof_in_hand
    tof_x = [point[0] for point in tof_in_hand]
    tof_y = [point[1] for point in tof_in_hand]
    tof_z = [point[2] for point in tof_in_hand]

    # Find the four nearest points in tof_in_hand
    nearest_points = []
    for j in range(len(tof_x)):
        if abs(tof_x[j] - x) <= 250 and abs(tof_y[j] - y) <= 250:
            nearest_points.append(tof_z[j])
        if len(nearest_points) == 4:
            break

    # Compute the average depth value of the four nearest points
    if len(nearest_points) > 0:
        mean_depth = np.mean(nearest_points)
        return mean_depth
    else:
        return 0

def get_depth_at_coord_wrist(x, y, tof_in_hand):
    if len(tof_in_hand) == 0:
        return None

    # 初始化最小距離和對應深度
    min_distance = float('inf')
    depth_at_min_distance = None

    # 遍歷所有點，找到距離 (x, y) 最近的一個點
    for point in tof_in_hand:
        point_x, point_y, point_z = point[0], point[1], point[2]
        distance = np.sqrt((point_x - x)**2 + (point_y - y)**2)
        
        if distance < min_distance:
            min_distance = distance
            depth_at_min_distance = point_z

    # 如果找到了最近的點，返回其深度值，否則返回 0
    if depth_at_min_distance is not None:
        return depth_at_min_distance
    else:
        return 0
    
def process_landmark(hand_landmarks, index, imgWidth, imgHeight, tof_in_hand, M, R, t):
    """
    Processes the landmark coordinates for a given index and returns the world coordinates.

    Parameters:
    hand_landmarks : List of hand landmarks.
    index : int
        Index of the landmark to process.
    imgWidth : int
        Width of the image.
    imgHeight : int
        Height of the image.
    tof_in_hand : object
        Depth information.
    M : ndarray
        Camera intrinsic matrix.
    R : ndarray
        Rotation matrix.
    t : ndarray
        Translation vector.

    Returns:
    tuple
        World coordinates (x, y, z) rounded to 3 decimal places.,z is calaculate from tof
    """
    lm = hand_landmarks[index]
    x = lm.x * imgWidth
    y = lm.y * imgHeight
    z = get_depth_at_coord(x, y, tof_in_hand)
    world_coor = image2World(x, y, z, M, R, t)
    
    x_world = round(world_coor[0, 0], 3)
    y_world = round(world_coor[1, 0], 3)
    
    return x_world, y_world,z

def adjust_z_for_pair(landmarks, real_hand_length, hand_line,  index, adjust_z, imgWidth, imgHeight, tof_in_hand, M, R, t, hand_average_z):
    z_low, z_high = hand_average_z - 5, hand_average_z + 16  # hand_average_z ± 16cm
    precision = 0.2  # Set precision
    i = hand_line[index][0]
    j = hand_line[index][1]
    
    target_length = real_hand_length[index]
    
    x1, y1, z1 = process_landmark(landmarks, i, imgWidth, imgHeight, tof_in_hand, M, R, t)
    z1 = adjust_z[i]
    # z1 remains unchanged
    while z_high - z_low > precision:
        z_mid = (z_low + z_high) / 2
        x2, y2, z2 = process_landmark(landmarks, j, imgWidth, imgHeight, tof_in_hand, M, R, t)
        z2 = z_mid  # Adjusted z value
        dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        if dist < target_length - precision:
            z_low = z_mid
        elif dist > target_length + precision:
            z_high = z_mid
        else:
            return z_mid
    return (z_low + z_high) / 2

def find_points_on_line_3d(x1, y1, z1, x2, y2, z2, I, J, K, dist, sign):
    # Direction vector of the line
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1

    # Vector from point (x1, y1, z1) to point (I, J, K)
    px = I - x1
    py = J - y1
    pz = K - z1

    # Dot product of direction vector and point vector
    dot_product = px * dx + py * dy + pz * dz

    # Magnitude of the direction vector squared
    direction_magnitude_squared = dx**2 + dy**2 + dz**2

    # Parameter t that minimizes the distance
    t = dot_product / direction_magnitude_squared

    # Closest point on the line
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    closest_z = z1 + t * dz

    # Shortest distance
    shortest_distance = math.sqrt((closest_x - I)**2 + (closest_y - J)**2 + (closest_z - K)**2)

    # If the shortest distance is greater than the specified distance
    if shortest_distance > dist:
        # print(f"The shortest distance ({shortest_distance}) is greater than the specified distance ({dist}). Returning the closest point on the line.")
        return round(closest_x, 3), round(closest_y, 3), round(closest_z, 3)

    # Normalizing the direction vector
    length = math.sqrt(dx**2 + dy**2 + dz**2)
    dx /= length
    dy /= length
    dz /= length

    # Quadratic equation coefficients
    A = dx**2 + dy**2 + dz**2
    B = 2 * (dx * (x1 - I) + dy * (y1 - J) + dz * (z1 - K))
    C = (x1 - I)**2 + (y1 - J)**2 + (z1 - K)**2 - dist**2

    # Discriminant
    discriminant = B**2 - 4 * A * C

    if discriminant < 0:
        return []  # No real solutions

    sqrt_discriminant = math.sqrt(discriminant)

    # Solutions for t
    t1 = (-B + sqrt_discriminant) / (2 * A)
    t2 = (-B - sqrt_discriminant) / (2 * A)

    # Corresponding points on the line
    point1 = (x1 + t1 * dx, y1 + t1 * dy, z1 + t1 * dz)
    point2 = (x1 + t2 * dx, y1 + t2 * dy, z1 + t2 * dz)

    # Choose the appropriate point based on the sign
    if sign >= 0:
        if t1 * dz > t2 * dz:
            chosen_point = point1
        else:
            chosen_point = point2
    else:
        if t1 * dz > t2 * dz:
            chosen_point = point2
        else:
            chosen_point = point1

    # Round the chosen point coordinates to three decimal places
    return round(chosen_point[0], 3), round(chosen_point[1], 3), round(chosen_point[2], 3)



    
    
def hand_tracking():
    tof = Tof()
    cap = cv2.VideoCapture(1)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils
    pTime = 0

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 創建 UDP 套接字
    serverAddressPort = ("127.0.0.1", 5052)  # 伺服器的 IP 位址和埠號
    
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
        # 內參
    fx = 1125.9
    # cx = 656.4
    cx = 640
    fy = 1127.2
    # cy = 363.5
    cy = 360

    # 相機矩陣
    M = np.array([[fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]])

    # 外參
    # 旋轉矩陣
    R = np.array([[0.999999, 0.001117, -0.000254],
                [-0.001116, 0.999997, -0.002008],
                [0.000255,  0.002009,  0.999999]])

    # 平移矩陣
    t = np.array([[1.5],
                [0.3],
                [0.0]])
    
    T = np.concatenate((R, t), axis=1)
    
    
    # store hand length
    real_hand_length = []
    # 順序:0-1, 1-2, 2-3, 3-4, 0-5, 5-6, 6-7, 7-8, 0-9, 9-10, 10-11, 11-12, 0-13, 13-14, 14-15, 15-16, 0-17, 17-18, 18-19, 19-20
    target_length = 20
    default_value = 0 

    real_hand_length.extend([default_value] * (target_length - len(real_hand_length)))
    
    hand_line = [
        [0, 1], [1, 2], [2, 3], [3, 4],
        [0, 5], [5, 6], [6, 7], [7, 8],
        [0, 9], [9, 10], [10, 11], [11, 12],
        [0, 13], [13, 14], [14, 15], [15, 16],
        [0, 17], [17, 18], [18, 19], [19, 20]
    ]
    
    start = False
    
    while True:
        ret, img = cap.read()
        if not ret:
            break
        median_range,real_distance,peak_rate = tof.get_frame()
        

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(imgRGB)

        imgHeight, imgWidth, _ = img.shape

        handArea,handDepth,hand_average_x,hand_average_y = handDetect(peak_rate, median_range)
        
        
        all_image_coords = []
        hull = []
        
        tof_in_hand = []
        
        
        # adjust_z store hand depth after adjust
        adjust_z = []
        target_length = 21
        default_value = 0  # 你希望填充的預設值，可以是 None、0 或其他值

        adjust_z.extend([default_value] * (target_length - len(adjust_z)))
        
        # start = False
        
        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
                
                hand_coords = [(lm.x * imgWidth, lm.y * imgHeight) for lm in handLms.landmark]
                
                hand_points = np.array(hand_coords, np.int32)
                hull = cv2.convexHull(hand_points)
                
                # 在影像上繪製凸包
                cv2.polylines(img, [hull], True, (0, 255, 0), 2)
                landMarkdepth = landmarkDepth(hand_coords, median_range,handArea)
                ###
                #rotation_x = calculate_rotation_x(handLms.landmark, imgWidth, imgHeight)
                #rotation_y = calculate_rotation_y(handLms.landmark)
                #rotation_z = calculate_rotation_z(handLms.landmark)
                rotation_x, rotation_y, rotation_z = calculate_rotation(handLms.landmark)
                
                # 計算手腕和中指指尖之間的距離(無單位)
                distance = calculate_distance(handLms.landmark,0,9)
                #real_distance
                # distance2 = calculate_distance(handLms.landmark,0,5)
                if(real_distance == 0):
                    align = 1
                else: align = real_distance / 300
                distance = distance * align
                
                rotation_z_distance = map_to_angle1(distance*100)
                rotation_z_angle= calculate_angle_between_vectors(hand_coords, 0, 5, 0, 17)
                rotation_z_angle = map_to_angle2(rotation_z_angle)
                
                rotation_z = fixRotationz_1(rotation_z_distance, rotation_z_angle, rotation_x)
                    
                
                
                # cv2.putText(img, f"Rotation X: {rotation_x:.2f}", (10, 50 + 30 * 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # cv2.putText(img, f"Rotation Y: {rotation_y:.2f}", (10, 50 + 30 * 7), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # cv2.putText(img, f"Rotation Z dist: {rotation_z_distance:.2f}", (10, 50 + 30 * 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # cv2.putText(img, f"Rotation Z angel: {rotation_z_angle:.2f}", (10, 50 + 30 * 9), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # cv2.putText(img, f"Rotation Z: {rotation_z:.2f}", (10, 50 + 30 * 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # cv2.putText(img, f"Rotation Z: {rotation_z:.2f}", (10, 50 + 30 * 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # cv2.putText(img, f"distance : {distance:.2f}", (10, 50 + 30 * 9), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # cv2.putText(img, f"real_distance : {real_distance:.2f}", (10, 50 + 30 * 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                #print(dist)
                # cv2.putText(img, f"distance : {distance2:.2f}", (10, 50 + 30 * 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                ###
                angles = hand_angle(hand_coords)
                #附加角度
                angles_str = ",".join([f"{angle:.2f}" for angle in angles])
                
                # for i, angle in enumerate(angles):
                #     #cv2.putText(img, f"Angle{i}: {float(angle)}", (10, 50 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                #     cv2.putText(img, f"Angle{i}: {angle:.2f}", (10, 50 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                # 傳送socketㄝ
                # data = []
                data = [0] * (len(handLms.landmark) * 3*2)
                # tof z 
                # 0~62 oldpoints 63~125 new points
                # for lm in handLms.landmark:
                #    data.extend([lm.x * imgWidth, imgHeight - lm.y * imgHeight, lm.z]) #3*21 = 63 / 0 - 62
                # for lm in handLms.landmark:
                #    data.extend([lm.x * imgWidth, imgHeight - lm.y * imgHeight, lm.z]) #3*21 = 63 / 0 - 62
                for i in range(8):
                        for j in range(8):
                            world_coor = tof2World_fov(i, j, median_range[i][j])
                            world_x = world_coor[0]
                            world_y = world_coor[1]
                            world_z = world_coor[2]
                            image_coor = world2Image(world_x, world_y, median_range[i][j], M, R, t, T)

                            if any(map(math.isnan, image_coor)):
                                continue

                            if 0 <= int(image_coor[0]) < imgWidth and 0 <= int(image_coor[1]) < imgHeight:
                                all_image_coords.append([int(image_coor[0]), int(image_coor[1]), median_range[j][i], j, i])
                            # 改用y, x, depth, i, j
                            for x_offset in range(-1, 2):
                                for y_offset in range(-1, 2):
                                    x = int(image_coor[0]) + x_offset
                                    y = int(image_coor[1]) + y_offset
                                    if 0 <= x < imgWidth and 0 <= y < imgHeight:
                                        img[y][x] = [0, 0, 255]
                hull_points = np.array(hull, np.int32)
                            
                if len(hull_points) > 0:
                    # 遍歷每一個需要判斷的點
                    for image_coor in all_image_coords:
                        # 使用 cv2.pointPolygonTest() 判斷點是否在凸包內
                        # 720 * 1280
                        x = image_coor[0]
                        y = image_coor[1]
                        z = image_coor[2]
                        i = image_coor[3]
                        j = image_coor[4]

                        dist = cv2.pointPolygonTest(hull_points, (x,y), False)
                        
                        # 根據返回值判斷點的位置
                        if dist > 0:
                            tof_in_hand.append([x, y, z, i, j])
                
                
                tof_in_hand = add_nearby_points1(all_image_coords, tof_in_hand, img, imgWidth, imgHeight)
                total_z = 0
                for x, y, z, i, j in tof_in_hand:
                # 在中心點標記白色
                    img[y][x] = (255, 255, 255)
                    total_z += z
                    # 在 3x3 的區域內標記白色
                    for x_offset in range(-1, 2):
                        for y_offset in range(-1, 2):
                            new_x = x + x_offset
                            new_y = y + y_offset
                            if 0 <= new_x < imgWidth and 0 <= new_y < imgHeight:
                                img[new_y][new_x] = (255, 255, 255)
                # hand_average_z for 平均手深度
                if len(tof_in_hand) > 0:
                    hand_average_z = total_z / len(tof_in_hand)
                else :
                    hand_average_z = 0
                # for oldpoints
                
                # adjust_z[0] = hand_average_z
                
                
                # real_hand_length
                # hand_line = [
                #     [0, 1], [1, 2], [2, 3], [3, 4],
                #     [0, 5], [5, 6], [6, 7], [7, 8],
                #     [0, 9], [9, 10], [10, 11], [11, 12],
                #     [0, 13], [13, 14], [14, 15], [15, 16],
                #     [0, 17], [17, 18], [18, 19], [19, 20]
                # ]
                
                # 這裡目標是得到各個landmark之間的長度
                # 並儲存在real_hand_length
                if cv2.waitKey(1) == ord('t'):
                    start = True
                    
                    # if len(tof_in_hand) > 0:
                    #     hand_average_z = total_z / len(tof_in_hand)
                        
                    # else :
                    #     and_average_z = 0
                    #     # for oldpoints
                    # adjust_z[0] = hand_average_z
                    
                    for index, pair in enumerate(hand_line):
                        i = pair[0]
                        j = pair[1]
                        # process_landmark 回傳世界坐標系的x,y,z, 其中 z 參考的是tof
                        x1, y1, z1 = process_landmark(handLms.landmark, i, imgWidth, imgHeight, tof_in_hand, M, R, t)
                        z1 = hand_average_z
                        x2, y2, z2 = process_landmark(handLms.landmark, j, imgWidth, imgHeight, tof_in_hand, M, R, t)
                        z2 = hand_average_z
                        # 計算兩點間距離   
                        dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
                        # print(dist)
                        real_hand_length[index] = round(dist,3)
                    print("depth:")
                    print(hand_average_z)
                    print(real_hand_length)
                
                # if start:
                #     for index, pair in enumerate(hand_line):
                
                # adjusted_z_for_pair0 = adjust_z_for_pair(handLms.landmark, real_hand_length, hand_line, 0, adjust_z, imgWidth, imgHeight, tof_in_hand, M, R, t, hand_average_z)
                

                
                
                wrist = handLms.landmark[0]
                x = wrist.x * imgWidth
                y = wrist.y * imgHeight
                # wristDepth = get_depth_at_coord(x, y, tof_in_hand)
                wristDepth = get_depth_at_coord_wrist(x, y, tof_in_hand)
                if wristDepth is None:
                    adjust_z[0] = 0
                else :
                    adjust_z[0] = wristDepth
                data[0] = x
                data[1] = y
                data[2] = wristDepth
                
                if wristDepth is None:
                    wristDepth = 0
                
                
                
                lm = handLms.landmark[1]
                x = lm.x * imgWidth
                # y = imgHeight - lm.y * imgHeight
                y = lm.y * imgHeight
                # z = landMarkdepth[i]   # 從 landMarkdepth 中獲得深度值
                # z = 10 + (lm.z * 100)
                z = wristDepth
                
                # data[i * 3] = round(world_coor1[0,0],3)
                # data[i * 3 + 1] = round(world_coor1[1,0],3)
                world_coor1 = image2World(x, y, z, M, R, t)
                x = round(world_coor1[0,0],3)
                y = round(world_coor1[1,0],3)
                
                text = f"coor1: ({x,y,z})"
                cv2.putText(img, text, (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, ), 2)
                
                x = lm.x * imgWidth
                # y = imgHeight - lm.y * imgHeight
                y = lm.y * imgHeight
                z = wristDepth+2
                world_coor1 = image2World(x, y, z, M, R, t)
                x = round(world_coor1[0,0],3)
                y = round(world_coor1[1,0],3)
                
                text = f"coor2: ({x,y,z})"
                cv2.putText(img, text, (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, ), 2)
                
                lm = handLms.landmark[0]
                x = lm.x * imgWidth
                # y = imgHeight - lm.y * imgHeight
                y = lm.y * imgHeight
                # z = landMarkdepth[i]   # 從 landMarkdepth 中獲得深度值
                # z = 10 + (lm.z * 100)
                z = wristDepth
                world_coor1 = image2World(x, y, z, M, R, t)
                x = round(world_coor1[0,0],3)
                y = round(world_coor1[1,0],3)
                text = f"wrist: ({x,y,z})"
                cv2.putText(img, text, (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, ), 2)
                
                
                # if start :
                #     index = 0
                #     pair = hand_line[index]
                    
                    
                #     first = pair[0]
                #     second = pair[1]
                #     lm1 = handLms.landmark[first]
                #     x0 = lm1.x * imgWidth
                #     y0 = lm1.y * imgHeight
                #     # adjust_z 放調整好的z
                #     z0 = adjust_z[first]

                #     world_coor = image2World(x0, y0, z0, M, R, t)
                #     I = round(world_coor[0, 0], 3)
                #     J = round(world_coor[1, 0], 3)
                #     K = adjust_z[first]
                    
                #     # x1, y1, z1 = process_landmark(handLms.landmark, pair[0], imgWidth, imgHeight, tof_in_hand, M, R, t)
                #     lm2 = handLms.landmark[second]
                #     x1 = lm2.x * imgWidth
                #     y1 = lm2.y * imgHeight
                #     z1 = wristDepth
                #     world_coor1 = image2World(x1, y1, z1, M, R, t)
                #     world_coor2 = image2World(x1, y1, z1 + 2, M, R, t)
                    
                #     x1 = round(world_coor1[0, 0], 3)
                #     y1 = round(world_coor1[1, 0], 3)
                #     z1 = wristDepth
                #     x2 = round(world_coor2[0, 0], 3)
                #     y2 = round(world_coor2[1, 0], 3)
                #     z2 = wristDepth + 2

                #     new_point = find_points_on_line_3d(x1, y1, z1, x2, y2, z2, I, J, K, real_hand_length[index],lm2.z)
                #     adjust_z[second] = new_point[2]
                #     text = f"new_coor 1: ({new_point[0],new_point[1],new_point[2]})"
                #     cv2.putText(img, text, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, ), 2)
                
                if start:
                    for index, pair in enumerate(hand_line):
                        
                        first = pair[0]
                        second = pair[1]
                        
                        # Processing first landmark
                        lm1 = handLms.landmark[first]
                        x0 = lm1.x * imgWidth
                        y0 = lm1.y * imgHeight
                        z0 = adjust_z[first]
                        
                        world_coor = image2World(x0, y0, z0, M, R, t)
                        I = round(world_coor[0, 0], 3)
                        J = round(world_coor[1, 0], 3)
                        K = adjust_z[first]
                        
                        # Processing second landmark
                        lm2 = handLms.landmark[second]
                        x1 = lm2.x * imgWidth
                        y1 = lm2.y * imgHeight
                        z1 = wristDepth
                        
                        tof_z = get_depth_at_coord(x1, y1, tof_in_hand)
                        world_coor1 = image2World(x1, y1, z1, M, R, t)
                        world_coor2 = image2World(x1, y1, z1 + 2, M, R, t)
                        

                        x1 = round(world_coor1[0, 0], 3)
                        y1 = round(world_coor1[1, 0], 3)
                        z1 = wristDepth
                        x2 = round(world_coor2[0, 0], 3)
                        y2 = round(world_coor2[1, 0], 3)
                        z2 = wristDepth + 2
                        
                        # Finding new point
                        # new_point = find_points_on_line_3d(x1, y1, z1, x2, y2, z2, I, J, K, real_hand_length[index], lm2.z)
                        # use tof z
                        
                        if tof_z is None:
                            tof_z = 0
                        new_point = find_points_on_line_3d(x1, y1, z1, x2, y2, z2, I, J, K, real_hand_length[index], tof_z - adjust_z[0])
                        
                        # 永遠往前
                        # new_point = find_points_on_line_3d(x1, y1, z1, x2, y2, z2, I, J, K, real_hand_length[index], 1)
                        
                        adjust_z[second] = new_point[2]
                    
                    text = f"new_z 5: ({ round(wristDepth + handLms.landmark[5].z*100,3),adjust_z[5]})"
                    cv2.putText(img, text, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, ), 2)
                    
                    text = f"new_z 6: ({ round(wristDepth + handLms.landmark[6].z*100,3),adjust_z[6]})"
                    cv2.putText(img, text, (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, ), 2)
                    
                    text = f"new_z 7: ({ round(wristDepth + handLms.landmark[7].z*100,3),adjust_z[7]})"
                    cv2.putText(img, text, (10, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, ), 2)
                    
                    text = f"new_z 8: ({ round(wristDepth + handLms.landmark[8].z*100,3),adjust_z[8]})"
                    cv2.putText(img, text, (10, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, ), 2)
                
                # use pure mediapipe
                # for i in range(len(handLms.landmark)):
                #     if i == 0:
                #         continue
                #     lm = handLms.landmark[i]
                #     x = lm.x * imgWidth
                #     # y = imgHeight - lm.y * imgHeight
                #     y = lm.y * imgHeight
                #     # z = landMarkdepth[i]   # 從 landMarkdepth 中獲得深度值
                #     # z = 10 + (lm.z * 100)
                #     z = wristDepth + (lm.z * 100)
                    
                #     world_coor1 = image2World(x, y, z, M, R, t)
                    
                #     # 校正後的x y
                #     # cal_x, cal_y = calibration(x/80.0, y/80.0, fixed_x, fixed_y)
                #     # z = get_depth_at_coord(x, y, tof_in_hand)
                #     # /10 for Convert to z in unity no need
                #     # z = median_range[cal_x][cal_y]  # 從 median_range 中獲得深度值
                #     data[i * 3] = round(world_coor1[0,0],3)
                #     data[i * 3 + 1] = round(world_coor1[1,0],3)
                #     data[i * 3 + 2] = z
                
                # new for adjusted_z
                for i in range(len(handLms.landmark)):
                    if i == 0:
                        continue
                    lm = handLms.landmark[i]
                    x = lm.x * imgWidth
                    # y = imgHeight - lm.y * imgHeight
                    y = lm.y * imgHeight
                    # z = landMarkdepth[i]   # 從 landMarkdepth 中獲得深度值
                    # z = 10 + (lm.z * 100)
                    z = adjust_z[i]
                    
                    world_coor1 = image2World(x, y, z, M, R, t)
                    
                    # 校正後的x y
                    # cal_x, cal_y = calibration(x/80.0, y/80.0, fixed_x, fixed_y)
                    # z = get_depth_at_coord(x, y, tof_in_hand)
                    # /10 for Convert to z in unity no need
                    # z = median_range[cal_x][cal_y]  # 從 median_range 中獲得深度值
                    data[i * 3] = round(world_coor1[0,0],3)
                    data[i * 3 + 1] = round(world_coor1[1,0],3)
                    data[i * 3 + 2] = z
                
                
                
                # lm = handLms.landmark[11]
                # x_11 = lm.x * imgWidth
                # # y = imgHeight - lm.y * imgHeight
                # y_11 = lm.y * imgHeight
                # z_11 = get_depth_at_coord(x_11, y_11, tof_in_hand)
                # world_coor1 = image2World(x_11, y_11, z_11, M, R, t)
                
                # x_11 = round(world_coor1[0,0],3)
                # y_11 = round(world_coor1[1,0],3)
                x_11, y_11, z_11 = process_landmark(handLms.landmark, 11, imgWidth, imgHeight, tof_in_hand, M, R, t)
                # 已是world_coor1
                
                # lm = handLms.landmark[12]
                # x_12 = lm.x * imgWidth
                # # y = imgHeight - lm.y * imgHeight
                # y_12 = lm.y * imgHeight
                # z_12 = get_depth_at_coord(x_12, y_12, tof_in_hand)
                # world_coor1 = image2World(x_12, y_12, z_12, M, R, t)
                
                # x_12 = round(world_coor1[0,0],3)
                # y_12 = round(world_coor1[1,0],3)
                
                # process_landmark 的 z 是參考tof的深度 這裡參考mediapipe的資訊
                x_12, y_12, z_12 = process_landmark(handLms.landmark, 12, imgWidth, imgHeight, tof_in_hand, M, R, t)
                
                x_0, y_0, z_0 = process_landmark(handLms.landmark, 0, imgWidth, imgHeight, tof_in_hand, M, R, t)
                z_0 = hand_average_z
                
                x_1, y_1, z_1 = process_landmark(handLms.landmark, 1, imgWidth, imgHeight, tof_in_hand, M, R, t)
                z_1 = hand_average_z
                if z_1 is not None:
                    dist_0_1 = math.sqrt(math.pow(x_0 - x_1, 2) + math.pow(y_0 - y_1, 2) + math.pow(z_0 - z_1, 2))
                else : 
                    dist_0_1 = 0
                
                if z_11 is not None:
                    dist_11_12 = math.sqrt(math.pow(x_11 - x_12, 2) + math.pow(y_11 - y_12, 2) + math.pow(z_11 - z_12, 2))
                else : 
                    dist_11_12 = 0
                
                text = f"hand avg depth: ({round((hand_average_z),3)})"
                cv2.putText(img, text, (10, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, ), 2)
                
                
                # x y 的校正係數
                fixed_x = [1.5,-2.8]
                fixed_y = [1.4,-0.3]
                # calibrationdepth = []
                # for newpoints
                # cal_x, cal_y = calibration(x/80.0, y/80.0, fixed_x, fixed_y)
                
                
                lm = handLms.landmark[0]
                x = lm.x * imgWidth
                y = lm.y * imgHeight
            
                z = get_depth_at_coord(x, y, tof_in_hand)
                
                # text = f"coord[0] depth: ({z})"
                # cv2.putText(img, text, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, ), 2)
                
                lm = handLms.landmark[12]
                x = lm.x * imgWidth
                y = lm.y * imgHeight
            
                z = get_depth_at_coord(x, y, tof_in_hand)
                
                # text = f"coord[12] depth: ({z})"
                # cv2.putText(img, text, (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, ), 2)
                
                for i in range(len(handLms.landmark)):
                    lm = handLms.landmark[i]
                    x = lm.x * imgWidth
                    # y = imgHeight - lm.y * imgHeight
                    y = lm.y * imgHeight
                    z = get_depth_at_coord(x, y, tof_in_hand)
                    world_coor1 = image2World(x, y, z, M, R, t)
                    
                    # 校正後的x y
                    # cal_x, cal_y = calibration(x/80.0, y/80.0, fixed_x, fixed_y)
                    # z = get_depth_at_coord(x, y, tof_in_hand)
                    # /10 for Convert to z in unity no need
                    # z = median_range[cal_x][cal_y]  # 從 median_range 中獲得深度值
                    data[63+i * 3] = round(world_coor1[0,0],3)
                    data[63+i * 3 + 1] = round(world_coor1[1,0],3)
                    data[63+i * 3 + 2] = z
                # data.extend(angles)  # 添加5個角度值 63-67
                # data.extend([rotation_x]) #points[68]
                # data.extend([rotation_y]) #points[69]                
                # # data.extend([rotation_z_distance]) #points[70]
                # data.extend([rotation_z]) #points[70]
                data_str = ",".join([str(val) for val in data])
                sock.sendto(str.encode(str(data)), serverAddressPort)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('img', img)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    hand_tracking()
        

        
