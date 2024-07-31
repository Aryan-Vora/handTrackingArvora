import datetime
import cv2
import mediapipe as mp

#TODO
# 1. ensure that there is only one click per mouse movemnt but the clicked state could be true for multiple frames
# 2. spikes should only go up once per click and not multiple times
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(2)

# Step 1: Initialize a list to store hand area values
hand_area_values = []

# Track total spikes and current spike state
total_spikes = 0
current_spike = False

# Function to detect spikes
def detect_spikes(values, threshold=0.2):
    global total_spikes, current_spike
    spikes = 0
    for i in range(1, len(values)):
        if values[i] > values[i - 1] * (1 + threshold):
            spikes += 1
            current_spike = True
        else:
            current_spike = False
    total_spikes = max(total_spikes, spikes)
    return spikes

# Function to determine the central grid cell
def get_grid_position(cx, cy, w, h):
    cell_width = w // 8
    cell_height = h // 8
    grid_x = min(cx // cell_width, 7)  # Ensure it doesn't go out of bounds
    grid_y = min(cy // cell_height, 7)  # Ensure it doesn't go out of bounds
    return grid_x, grid_y

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w, _ = image.shape
    grid_positions = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            min_x, min_y = w, h
            max_x = max_y = 0
            
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                min_x, min_y = min(min_x, cx), min(min_y, cy)
                max_x, max_y = max(max_x, cx), max(max_y, cy)
                
                cv2.putText(image, f'{id}: ({cx}, {cy})', (cx, cy), cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 0, 255), 1)
            
            bbox_width, bbox_height = max_x - min_x, max_y - min_y
            hand_area = bbox_width * bbox_height
            
            # Step 2: Update the list with the new hand area value
            hand_area_values.append(hand_area)
            
            # Step 3: Limit the size of the list
            if len(hand_area_values) > 100:
                hand_area_values.pop(0)
            
            # Step 4: Draw the graph
            graph_height = 100
            graph_width = 100  # Display only the last 100 values
            base_line = 110  # Position the graph at the top left of the image
            for i in range(len(hand_area_values)):
                area = hand_area_values[i]
                normalized_area = int((area / max(hand_area_values)) * graph_height)  # Normalize the area value
                cv2.line(image, (i, base_line), (i, base_line - normalized_area), (0, 255, 0), 1)
                
            # Draw the text annotations
            cv2.putText(image, 'Hand Size', (10, base_line - graph_height - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
            
            # Detect spikes and draw the spike count
            spike_count = detect_spikes(hand_area_values)
            # Adjusted code to move the text to the bottom left
            text = f'Spikes: {total_spikes}'
            text_width, text_height = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            text_x = 10  # 10 pixels from the left edge
            text_y = image.shape[0] - 40  # 40 pixels from the bottom (leaving space for "Clicks" text)
            cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            
            # Draw the clicked status
            clicked_status = 'true' if current_spike else 'false'
            clicks_text = f'Clicks: {clicked_status}'
            clicks_text_width, clicks_text_height = cv2.getTextSize(clicks_text, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            clicks_text_x = 10  # 10 pixels from the left edge
            clicks_text_y = image.shape[0] - 20  # 20 pixels from the bottom
            cv2.putText(image, clicks_text, (clicks_text_x, clicks_text_y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            
            # Determine the central grid cell of the hand bounding box
            central_x = (min_x + max_x) // 2
            central_y = (min_y + max_y) // 2
            grid_x, grid_y = get_grid_position(central_x, central_y, w, h)
            grid_positions.append((grid_x, grid_y))

    # Check if there is a spike and print the grid position
    if current_spike and grid_positions:
        # Find the most central grid position
        central_grid_x, central_grid_y = grid_positions[len(grid_positions) // 2]
        print(f"Click on {chr(65 + central_grid_y)}x{central_grid_x + 1} at {central_x}, {central_y} at time {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")

    # Draw the 5x5 grid
    for i in range(1, 8):
        cv2.line(image, (i * w // 8, 0), (i * w // 8, h), (255, 255, 255), 1)
        cv2.line(image, (0, i * h // 8), (w, i * h // 8), (255, 255, 255), 1)

    cv2.imshow('Hand Tracking', image)
    if cv2.waitKey(10) & 0xFF == ord('q') or cv2.getWindowProperty('Hand Tracking', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
