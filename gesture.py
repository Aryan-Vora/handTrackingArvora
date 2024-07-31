import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Define gesture detection function
def determine_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y

    thumb_is_open = thumb_tip < thumb_ip
    index_is_open = index_tip < index_pip
    middle_is_open = middle_tip < middle_pip
    ring_is_open = ring_tip < ring_pip
    pinky_is_open = pinky_tip < pinky_pip

    if thumb_is_open and index_is_open and middle_is_open and ring_is_open and pinky_is_open:
        return "Open Hand"
    if not thumb_is_open and not index_is_open and not middle_is_open and not ring_is_open and not pinky_is_open:
        return "Fist"
    if thumb_is_open and not index_is_open and not middle_is_open and not ring_is_open and not pinky_is_open:
        # Additional check to ensure thumb is significantly higher than other fingers
        if thumb_tip < index_tip and thumb_tip < middle_tip and thumb_tip < ring_tip and thumb_tip < pinky_tip:
            return "Thumbs Up"
    if not thumb_is_open and index_is_open and not middle_is_open and not ring_is_open and not pinky_is_open:
        return "Pointing"
    return "Unknown"

# Set up video capture
cap = cv2.VideoCapture(2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w, _ = image.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Calculate bounding box
            min_x, min_y = w, h
            max_x, max_y = 0, 0
            hand_points = []
            
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                min_x, min_y = min(min_x, cx), min(min_y, cy)
                max_x, max_y = max(max_x, cx), max(max_y, cy)
                hand_points.append([cx, cy])
                
                cv2.putText(image, f'{id}: ({cx}, {cy})', (cx, cy), cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 0, 255), 1)
            
            # Draw convex hull
            hand_points = np.array(hand_points)
            hull = cv2.convexHull(hand_points)
            cv2.polylines(image, [hull], True, (0, 255, 0), 2)

            # Iterate over each detected hand
            gestures = []
            for hand_landmarks in results.multi_hand_landmarks:
                # Detect gesture for each hand
                gesture = determine_gesture(hand_landmarks)
                gestures.append(gesture)

            # Display the list of gestures on the image
            gesture_text = ', '.join(gestures)
            cv2.putText(image, gesture_text, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cv2.imshow('Hand Tracking', image)
    if cv2.waitKey(10) & 0xFF == ord('q') or cv2.getWindowProperty('Hand Tracking', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
