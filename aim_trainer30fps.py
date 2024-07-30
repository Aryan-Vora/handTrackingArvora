import cv2
import mediapipe as mp
import math
import socket
import numpy as np
import serial
import serial.tools.list_ports
import struct
import time


HEIGHT = 1280
WIDTH = 720


class Tof:

    def __init__(self, port="COM3"):
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
        except serial.SerialException:
            print(f"Failed to open port: {self.port}.")
            print("Available ports:")
            for port, desc, hwid in sorted(serial.tools.list_ports.comports()):
                print(f"{port}: {desc} [{hwid}]")
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
    

    def _get_response(self) -> None:
        """Catch response after command is sent."""
        try:
            response = bytearray()
            while True:
                char = self.serial.read(1)
                if char == b'\r':
                    continue
                elif char == b'\n':
                    break
                response.extend(char)
            try:
                decoded_response = response.decode('ASCII')
            except UnicodeDecodeError as e:
                print(f"Decoding error at byte: {response[e.start:e.end].hex()} ({response[e.start:e.end]})")
                raise e
            print(f"[RESPONSE] {decoded_response}")
        except Exception as e:
            print(f"An error occurred: {e}")
            #end the program and run it again
            exit(1)


    def get_frame(self) -> float:
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
        self.buffer += self.serial.read(size=ind)
        if len(self.buffer) < 593:
            return (None, None)
    
        # Parse signal data
        peak_rate_kcps_per_spad = np.zeros((8, 8), dtype=np.uintc)
        for i in range(64):
            peak_rate_kcps_per_spad[i // 8][i % 8] = struct.unpack("<I", self.buffer[9 * i + 11:9 * i + 15])[0]
    
        # Parse median range data
        median_range = np.zeros((8, 8), dtype=np.short)
        for i in range(64):
            median_range[i // 8][i % 8] = struct.unpack("<h", self.buffer[9 * i + 15:9 * i + 17])[0]
            median_range[i // 8][i % 8] /= 40.0  # Convert to cm
    
        # Filter valid depths within the specified range
        min_dist, max_dist = 10, 50
        valid_depths = median_range[(median_range >= min_dist) & (median_range <= max_dist)]
        average_depth = np.mean(valid_depths) if valid_depths.size > 0 else 0
    
        # Clear buffer
        self.buffer = bytearray()
    
        return median_range, average_depth, peak_rate_kcps_per_spad

def calculate_distance(hand_landmarks, i, j):
    x1, y1 = hand_landmarks[i].x, hand_landmarks[i].y
    x2, y2 = hand_landmarks[j].x, hand_landmarks[j].y
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def handDetect(peak_rate, median_range):
    # This function detects hand presence based on peak rate and median range
    # and calculates the average valid median range.
    result = [[False]*8 for _ in range(8)]
    valid_median_range, valid_count, x, y = 0, 0, 0, 0

    for i in range(8):
        for j in range(8):
            if peak_rate[i][j] >= 60000 and 10 <= median_range[i][j] <= 50:
                result[i][j] = True
                valid_median_range += median_range[i][j]
                valid_count += 1
                x += i
                y += j

    avg_valid_median_range = valid_median_range / valid_count if valid_count > 0 else 0
    return result, avg_valid_median_range, x / valid_count, y / valid_count

def world2Image(x, y, z, M, R, t, T):
    # Convert image coordinates to a ray in the camera coordinate system
    # imagePoint = np.array([x, y, z])
    imagePoint = np.array([x, y, z, 1])
    
    if z == 0:
        image_coor = [0, 0, 0, 0]
    else:
        image_coor = imagePoint / z
    # image_coor = M.dot(image_coor)
    # image_coor = R.dot(image_coor)
    # print("image coor", image_coor)
    return M.dot(T.dot(image_coor))

def image2World(u, v, zConst, M, R, t):
    if u is None or v is None or zConst is None:
        return np.array([[0], [0], [0]])
    # Convert image coordinates to a ray in the camera coordinate system
    imagePoint = np.array([[u], [v], [1]])

    # Convert the ray to a point in the camera coordinate system
    rightMatrix = np.linalg.inv(R).dot(np.linalg.inv(M).dot(imagePoint))

    # Calculate the ray corresponding to the depth
    leftMatrix = np.linalg.inv(R).dot(t)
    s = (zConst + leftMatrix[2]) / rightMatrix[2]

    # Apply the depth to the ray and convert it to a point in the world coordinate system
    cameraPoint = np.linalg.inv(R).dot(s * np.linalg.inv(M).dot(imagePoint) - t)

    return cameraPoint

def tof2World_fov(x, y, D):
    # This function converts ToF sensor coordinates to world coordinates using field of view angles.
    FOV_h_rad, FOV_v_rad = map(math.radians, [22.5, 22.5])
    theta_h = ((x - 3.5) / 3.5) * FOV_h_rad
    theta_v = ((y - 3.5) / 3.5) * FOV_v_rad
    return D * math.tan(theta_h), D * math.tan(theta_v), D

def add_nearby_points1(all_image_coords, tof_in_hand, img):
    # This function filters points based on depth range and adds nearby points within a 3x3 grid if they fall within the same depth range.
    median_depth = np.median([point[2] for point in tof_in_hand])
    depth_range = (median_depth - 7, median_depth + 7)

    # Filter tof_in_hand points to include only those within the depth range
    filtered_tof = [point for point in tof_in_hand if depth_range[0] <= point[2] <= depth_range[1]]
    median_depth = np.median([point[2] for point in filtered_tof])
    cv2.putText(img, f"median depth: ({median_depth})", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    for image_coor in [coor for coor in all_image_coords if coor not in filtered_tof]:
        x, y, z, i, j = image_coor
        # Check if the depth of the current coordinate is within the depth range
        if depth_range[0] <= z <= depth_range[1]:
            # Generate neighboring points within a 3x3 grid
            neighbors = [(i + dx, j + dy) for dx in range(-1, 2) for dy in range(-1, 2)]
            
            # Check if any neighboring point is within the depth range and add to filtered_tof if true
            for ni, nj in neighbors:
                if 0 <= ni < 8 and 0 <= nj < 8 and any(depth_range[0] <= pt[2] <= depth_range[1] for pt in filtered_tof if pt[3] == ni and pt[4] == nj):
                    filtered_tof.append([x, y, z, i, j])
                    break

    return filtered_tof

def get_depth_at_coord(x, y, tof_in_hand):
    if not tof_in_hand:
        return None
    # Calculate the Euclidean distance from the given (x, y) coordinates to each point in tof_in_hand
    distances = [(np.sqrt((px - x) ** 2 + (py - y) ** 2), pz) for px, py, pz in [(p[0], p[1], p[2]) for p in tof_in_hand]]
    
    distances.sort(key=lambda d: d[0])
    nearest_points = [d[1] for d in distances[:4]]
    return np.mean(nearest_points) if nearest_points else 0

def get_depth_at_coord_wrist(x, y, tof_in_hand):
    if len(tof_in_hand) == 0:
        return None

    # Initialize the minimum distance and corresponding depth
    min_distance = float('inf')
    depth_at_min_distance = None

    # Iterate through all points to find the one closest to (x, y)
    for point in tof_in_hand:
        point_x, point_y, point_z = point[0], point[1], point[2]
        distance = np.sqrt((point_x - x)**2 + (point_y - y)**2)
        
        if distance < min_distance:
            min_distance = distance
            depth_at_min_distance = point_z

    # If the closest point is found, return its depth value, otherwise return 0
    if depth_at_min_distance is not None:
        return depth_at_min_distance
    else:
        return 0
    

def process_landmark(hand_landmarks, index, imgWidth, imgHeight, tof_in_hand, M, R, t):
    lm = hand_landmarks[index]
    # Convert normalized landmark coordinates to image coordinates
    x, y = lm.x * imgWidth, lm.y * imgHeight

    z = get_depth_at_coord(x, y, tof_in_hand)
    world_coor = image2World(x, y, z, M, R, t)
    return round(world_coor[0, 0], 3), round(world_coor[1, 0], 3), z

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

    # Find length of direction vector
    line_length = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    # Projection factor to find the point on the line closest to (I, J, K)
    projection_factor = dot_product / (line_length ** 2)

    # Calculate the closest point on the line
    closest_x = x1 + projection_factor * dx
    closest_y = y1 + projection_factor * dy
    closest_z = z1 + projection_factor * dz

    # Direction vector from the closest point to (I, J, K)
    dir_x = I - closest_x
    dir_y = J - closest_y
    dir_z = K - closest_z

    # Normalize the direction vector
    dir_length = math.sqrt(dir_x ** 2 + dir_y ** 2 + dir_z ** 2)
    dir_x /= dir_length
    dir_y /= dir_length
    dir_z /= dir_length

    # Move along the direction vector to the desired distance
    new_x = closest_x + sign * dist * dir_x
    new_y = closest_y + sign * dist * dir_y
    new_z = closest_z + sign * dist * dir_z

    return new_x, new_y, new_z

def hand_tracking():
    tof = Tof()
    
    # Ensure you get the right camera
    cap = cv2.VideoCapture(1)
    
    # Initialize MediaPipe Hands
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    pTime = 0

    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    serverAddressPort = ("127.0.0.1", 12345)  # Server IP address and port number
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, HEIGHT)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WIDTH)

    # Camera intrinsic parameters
    fx = 1125.9
    cx = 640
    fy = 1127.2
    cy = 360

    # Camera matrix
    M = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])

    # Extrinsic parameters
    # Rotation matrix
    R = np.array([[0.999999, 0.001117, -0.000254],
                  [-0.001116, 0.999997, -0.002008],
                  [0.000255,  0.002009,  0.999999]])

    # Translation matrix
    t = np.array([[1.5],
                  [0.3],
                  [0.0]])
    
    # Combine rotation and translation into a single transformation matrix
    T = np.concatenate((R, t), axis=1)

    # Order: 0-1, 1-2, 2-3, 3-4, 0-5, 5-6, 6-7, 7-8, 0-9, 9-10, 10-11, 11-12, 0-13, 13-14, 14-15, 15-16, 0-17, 17-18, 18-19, 19-20
    default_value = 0
    frame_count = 0
    frame_generated = 0
    depth_update_count = 0  # Counter for depth update synchronization
    # Store hand length
    input_display = np.zeros((WIDTH, HEIGHT, 3), dtype=np.uint8)
    
    # List of hand distances for each frame capped at last 120 frames
    hand_distances = [[0, 0, 0]]
    points = 0
    existing = False
    
    while True:
        ret, img = cap.read()
        if not ret:
            break
        input_display.fill(0)
        frame_count += 1
        depth_update_count += 1

        # Get frame data from ToF sensor every 2nd frame (15fps)
        if depth_update_count % 2 == 0:
            median_range, real_distance, peak_rate = tof.get_frame()
            depth_update_count = 0

        # Convert image to RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(imgRGB)

        imgHeight, imgWidth, _ = img.shape        

        all_image_coords = []
        hull = []
        tof_in_hand = []

        # Adjusted hand depth
        adjust_z = [default_value] * 21
        
        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:

                # Get hand coordinates
                hand_coords = [(lm.x * imgWidth, lm.y * imgHeight) for lm in handLms.landmark]
                
                # Convert hand coordinates to numpy array
                hand_points = np.array(hand_coords, np.int32)
                
                # Calculate the convex hull of the hand points
                hull = cv2.convexHull(hand_points)

                # Calculate the distance between the wrist and the tip of the middle finger (unitless)
                distance = calculate_distance(handLms.landmark, 0, 9)
                
                # Align distance based on real distance
                align = 1 if real_distance == 0 else real_distance / 300
                distance *= align
                
                # Initialize data array
                data = [0] * (len(handLms.landmark) * 3 * 2)
                
                for i in range(8):
                    for j in range(8):
                        # Convert ToF sensor data to world coordinates
                        world_coor = tof2World_fov(i, j, median_range[i][j])
                        world_x, world_y, world_z = world_coor
                        
                        # Convert world coordinates to image coordinates
                        image_coor = world2Image(world_x, world_y, median_range[i][j], M, R, t, T)
                        
                        if any(map(math.isnan, image_coor)):
                            continue
                        
                        if 0 <= int(image_coor[0]) < imgWidth and 0 <= int(image_coor[1]) < imgHeight:
                            all_image_coords.append([int(image_coor[0]), int(image_coor[1]), median_range[j][i], j, i])
                        
                        # Mark the image coordinates on the image
                        for x_offset in range(-1, 2):
                            for y_offset in range(-1, 2):
                                x = int(image_coor[0]) + x_offset
                                y = int(image_coor[1]) + y_offset
                                if 0 <= x < imgWidth and 0 <= y < imgHeight:
                                    img[y][x] = [0, 0, 255]
                
                # Convert hull points to numpy array
                hull_points = np.array(hull, np.int32)    
                if len(hull_points) > 0:
                    # Iterate through each point that needs to be judged
                    for image_coor in all_image_coords:
                        # Use cv2.pointPolygonTest() to determine if the point is inside the convex hull
                        x = image_coor[0]
                        y = image_coor[1]
                        z = image_coor[2]
                        i = image_coor[3]
                        j = image_coor[4]

                        dist = cv2.pointPolygonTest(hull_points, (x, y), False)
                        
                        # Determine the position of the point based on the return value
                        if dist > 0:
                            tof_in_hand.append([x, y, z, i, j])

                # Add nearby points to tof_in_hand
                tof_in_hand = add_nearby_points1(all_image_coords, tof_in_hand, img)
                total_z = 0

                for x, y, z, i, j in tof_in_hand:
                    # Mark the center point in white
                    img[y][x] = (255, 255, 255)
                    total_z += z
                    # Mark a 3x3 area in white
                    for x_offset in range(-1, 2):
                        for y_offset in range(-1, 2):
                            new_x = x + x_offset
                            new_y = y + y_offset
                            if 0 <= new_x < imgWidth and 0 <= new_y < imgHeight:
                                img[new_y][new_x] = (255, 255, 255)

                # Calculate the average hand depth
                if len(tof_in_hand) > 0:
                    hand_average_z = total_z / len(tof_in_hand)
                else:
                    hand_average_z = hand_distances[-1][2]

                for i in range(len(handLms.landmark)):
                    if i == 0:
                        continue  # Skip the first landmark
                    lm = handLms.landmark[i]
                    x = lm.x * imgWidth
                    y = lm.y * imgHeight
                    z = adjust_z[i]

                    # Convert image coordinates to world coordinates
                    world_coor1 = image2World(x, y, z, M, R, t)

                    # Store the world coordinates in the data array
                    data[i * 3] = round(world_coor1[0,0],3)
                    data[i * 3 + 1] = round(world_coor1[1,0],3)
                    data[i * 3 + 2] = z

                # Trim hand_distances to the last 120 frames
                hand_distances = hand_distances[-120:]
                
                # Get depth at the coordinates of the first landmark and update z
                lm = handLms.landmark[0]
                x = lm.x * imgWidth
                y = lm.y * imgHeight
                z = get_depth_at_coord(x, y, tof_in_hand)

                # Get depth at the coordinates of the 12th landmark and update z
                lm = handLms.landmark[12]
                x = lm.x * imgWidth
                y = lm.y * imgHeight
                z = get_depth_at_coord(x, y, tof_in_hand)
            
                for i in range(len(handLms.landmark)):
                    lm = handLms.landmark[i]
                    x = lm.x * imgWidth  # Convert normalized x-coordinate to image coordinate
                    y = lm.y * imgHeight  # Convert normalized y-coordinate to image coordinate
                    z = get_depth_at_coord(x, y, tof_in_hand)  # Get depth value at the given coordinates

                    # Convert image coordinates to world coordinates
                    world_coor1 = image2World(x, y, z, M, R, t)
                    
                    # Store the world coordinates in the data array
                    data[63 + i * 3] = round(world_coor1[0, 0], 3)
                    data[63 + i * 3 + 1] = round(world_coor1[1, 0], 3)
                    data[63 + i * 3 + 2] = z

                sock.sendto(str.encode(str(data)), serverAddressPort)
        
                # Display the average depth of the hand on the image
                # text = f"hand avg depth: ({round((hand_average_z),3)})"
                # cv2.putText(input_display, text, (10, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                hand_distances.append((x,y, hand_average_z))

        # Update ball position every frame (30fps)
        if len(hand_distances) > 0 and hand_distances[-1][2] != 0:
            # Calculate the size of the dot inversely proportional to the depth
            depth = hand_distances[-1][2]
            # dot_size = max(1, int(1000 / depth)) 
            dot_size = 25
            cv2.circle(input_display, (int(hand_distances[-1][0]), int(hand_distances[-1][1])), dot_size, (0, 255, 0), -1)
        
        # Simple game where a square is generated every 20th frame and the user has to click on it 
        # Generate every 20 frames if there is not an already existing square
        if frame_count % 20 == 0 and not existing:
            target_x = np.random.randint(0, HEIGHT-200)
            target_y = np.random.randint(0, WIDTH-200)
            frame_generated = frame_count
            existing = True

        # Check if the user has clicked on the square and measure the frames between the square generation and the click
        # If the click is within the square, increment the points by 100 - frames between generation and click
        if existing:
            cv2.rectangle(input_display, (target_x, target_y), (target_x + 50, target_y + 50), (0, 0, 255), -1)
            if hand_distances[-1][0] > target_x and hand_distances[-1][0] < target_x + 75 and hand_distances[-1][1] > target_y and hand_distances[-1][1] < target_y + 75:
                if frame_count - frame_generated < 100:
                    points += 100 - (frame_count - frame_generated)
                existing = False        

        # Display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        #cv2.putText(img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(input_display, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show point counter
        cv2.putText(input_display, f"Points: {points}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('img', input_display)
        
        # Exit the program
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("img", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    hand_tracking()