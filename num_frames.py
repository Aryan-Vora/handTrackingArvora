import cv2
import time

# Start video capture
cap = cv2.VideoCapture(1)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Capture frames for a certain period to calculate FPS
num_frames = 120
print("Capturing {0} frames...".format(num_frames))

# Start time
start = time.time()

# Grab a few frames
for i in range(0, num_frames):
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

# End time
end = time.time()

# Time elapsed
seconds = end - start
fps = num_frames / seconds

print("Estimated frames per second : {0}".format(fps))

# Release video capture
cap.release()
cv2.destroyAllWindows()