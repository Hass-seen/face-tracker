import cv2
import json

# Read the video
video = cv2.VideoCapture("video.mp4")

# Check if the video was opened successfully
if not video.isOpened():
    print("Error opening video file")
    exit()

# Initialize a list to store the tracking data
tracking_data = []


reducer=0
# Read each frame of the video
while True:
    success, frame = video.read()

    # Check if the frame was read successfully
    if not success:
        break

    # Convert the frame to HSV color space (for object tracking) or grayscale (for face tracking)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range of colors to detect (for object tracking)
    lower_color = (0, 0, 0)
    upper_color = (180, 255, 50)

    # Threshold the HSV image to get only the desired colors (for object tracking)
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Load a pre-trained face detection model (for face tracking)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    
    # Convert the frame to grayscale (for face tracking)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame (for face tracking)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Check if any faces or objects were detected
    if len(faces) == 0:
        print("No faces or objects detected")
        break

    # Select the first face or object
    face = faces[0]

    # Extract the face or object coordinates
    x, y, w, h = face

    # Use the MeanShift algorithm to track the face or object
    track_window = cv2.meanShift(gray[y:y+h, x:x+w], (x, y, w, h), criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1))

    # Extract the tracking window coordinates
    cx, cy = track_window

    # Draw a rectangle around the tracked face or object
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Store the tracking data
    if reducer %5==0:
       tracking_data.append((cx, cy))
    reducer+=1

# Release the video capture
video.release()

# Save the tracking data to a file
track= [tr[1] for tr in tracking_data ]
JasonOBJ = json.dumps(track, indent=2)
with open("track.json", "w") as outfile:
        outfile.write(JasonOBJ)