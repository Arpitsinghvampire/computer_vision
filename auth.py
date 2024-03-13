'''
import cv2
import mediapipe as mp
import numpy as np
import threading
import pyautogui as py
frame_width = 640
frame_height = 480


def eye_motion(left_eye,right_eye):
    left = left_eye
    right=right_eye
                        
    for landmark in left:
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)
        cv2.circle(frame, (x, y), 3, (0, 255, 255))
        
    for landmark in right:
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)
        cv2.circle(frame, (x, y), 3, (0, 255, 255))
    print("Right: ", (right[0].y - right[1].y))
    print("Left: ", (left[0].y - left[1].y))
    if (left[0].y - left[1].y) < 0.004 and (right[0].y - right[1].y) > 0.015:
        py.leftClick()#500,850,1,1,
            
    if (right[0].y - right[1].y) < 0.004 and (left[0].y - left[1].y) > 0.015:
        py.rightClick()#500,850,1,1,   
            # py.hotkey('winkey','H')


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Initialize Mediapipe Face Detection module
face_detection =face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Check if the face_detection object is successfully initialized
if not face_detection:
    print("Error: Failed to initialize FaceDetection module.")
    exit()

# Capture the initial frame to get the root user's image
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        print("Failed to capture initial frame.")
        break

    cv2.imshow('Capture Root User Image', frame)

    # Press 'Space' key to capture the root user's image
    if cv2.waitKey(1) & 0xFF == ord(' '):
        root_user_image = frame.copy()
        break

cap.release()
cv2.destroyAllWindows()

# Process the root user image to get the facial landmarks
with mp.solutions.face_mesh.FaceMesh(refine_landmarks=True) as face_detection:
    root_user_results = face_detection.process(cv2.cvtColor(root_user_image, cv2.COLOR_BGR2RGB))
    landmark_points = root_user_results.multi_face_landmarks
    root_user_landmarks = landmark_points[0].landmark
    root_user_landmarks_check=[root_user_landmarks[1],root_user_landmarks[468],root_user_landmarks[473],root_user_landmarks[234],root_user_landmarks[454],root_user_landmarks[14]]

            
print(root_user_landmarks[1])
# Release the FaceDetection module before entering the real-time loop
face_detection = None

# Initialize the tracker
tracker = cv2.TrackerCSRT_create()
tracking_initiated = False

# Set the frame width and height for performance improvement

cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Define a threshold for landmark matching
landmark_threshold = 0.2

# Run real-time face detection with bounding box drawing and tracking
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Resize the frame for faster processing
    frame = cv2.resize(frame, (frame_width, frame_height))

    try:
        # Process the frame with face detection
        with mp.solutions.face_mesh.FaceMesh(refine_landmarks=True) as face_detection:
            results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            landmark_points = results.multi_face_landmarks
            current_face_landmarks = landmark_points[0].landmark

        # Draw bounding box around the detected face(s)
        if current_face_landmarks:
            # for detection in results.detections:
            #     bboxC = detection.location_data.relative_bounding_box
            #     ih, iw, _ = frame.shape
            #     bbox = (
            #         int(bboxC.xmin * iw), int(bboxC.ymin * ih),
            #         int(bboxC.width * iw), int(bboxC.height * ih)
            #     )

                # Compare the landmarks of the detected face with the root user landmarks
                current_face_landmarks_check=[current_face_landmarks[1],current_face_landmarks[468],current_face_landmarks[473],current_face_landmarks[234],current_face_landmarks[454],current_face_landmarks[14]]
                landmarks_distance = np.mean(np.sqrt(
                    (current_face_landmarks_check[0].x - root_user_landmarks_check[0].x) ** 2 +
                    (current_face_landmarks_check[0].y - root_user_landmarks_check[0].y) ** 2
                ))
                
                # If the landmarks are close, consider it a match
                if landmarks_distance < landmark_threshold:
                    # for i in current_face_landmarks_check:
                        left = [current_face_landmarks[145], current_face_landmarks[159]]
                        right=[current_face_landmarks[374], current_face_landmarks[386]]
                        cv2.circle(frame, (int(current_face_landmarks[1].x*frame_width), int(current_face_landmarks[1].y*frame_height)), 3, (0, 255, 255))
                        cv2.circle(frame, (int(current_face_landmarks[145].x*frame_width), int(current_face_landmarks[145].y*frame_height)), 3, (0, 255, 255))
                        cv2.circle(frame, (int(current_face_landmarks[159].x*frame_width), int(current_face_landmarks[159].y*frame_height)), 3, (0, 255, 255))
                        cv2.circle(frame, (int(current_face_landmarks[374].x*frame_width), int(current_face_landmarks[374].y*frame_height)), 3, (0, 255, 255))
                        cv2.circle(frame, (int(current_face_landmarks[386].x*frame_width), int(current_face_landmarks[386].y*frame_height)), 3, (0, 255, 255))
                        print("Nose\n",current_face_landmarks[1])
                        print("Left Eye",current_face_landmarks[145],current_face_landmarks[159])
                        print("Right Eye",current_face_landmarks[374],current_face_landmarks[386])
                threading.Thread(target=eye_motion,args=(left,right)).start()

    except Exception as e:
        import traceback
        traceback.print_exc()  # Print the entire exception traceback
        print(f"Error during face detection: {e}")

    cv2.imshow('Real-time Face Detection', frame)

    # Break the loop if 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
'''
'''
import cv2
import mediapipe as mp
import numpy as np
import threading
import pyautogui as py

frame_width = 640
frame_height = 480


def eye_motion(left_eye, right_eye):
    left = left_eye
    right = right_eye

    for landmark in left:
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)
        cv2.circle(frame, (x, y), 3, (0, 255, 255))

    for landmark in right:
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)
        cv2.circle(frame, (x, y), 3, (0, 255, 255))

    print("Right: ", (right[0].y - right[1].y))
    print("Left: ", (left[0].y - left[1].y))

    if (left[0].y - left[1].y) < 0.004 and (right[0].y - right[1].y) > 0.015:
        py.click(button='left')  # Perform left click

    if (right[0].y - right[1].y) < 0.004 and (left[0].y - left[1].y) > 0.015:
        py.click(button='right')  # Perform right click
        

def cursor(current_face_landmarks[1]):
    x,y,z=[float(a) for a in current__face_landmarks[1]]        
        
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Initialize Mediapipe Face Detection module
face_detection = face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Check if the face_detection object is successfully initialized
if not face_detection:
    print("Error: Failed to initialize FaceDetection module.")
    exit()

# Capture the initial frame to get the root user's image
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        print("Failed to capture initial frame.")
        break

    cv2.imshow('Capture Root User Image', frame)

    # Press 'Space' key to capture the root user's image
    if cv2.waitKey(1) & 0xFF == ord(' '):
        root_user_image = frame.copy()
        break

cap.release()
cv2.destroyAllWindows()

# Process the root user image to get the facial landmarks
with mp.solutions.face_mesh.FaceMesh(refine_landmarks=True) as face_detection:
    root_user_results = face_detection.process(cv2.cvtColor(root_user_image, cv2.COLOR_BGR2RGB))
    landmark_points = root_user_results.multi_face_landmarks
    root_user_landmarks = landmark_points[0].landmark
    root_user_landmarks_check = [root_user_landmarks[1], root_user_landmarks[468], root_user_landmarks[473],
                                  root_user_landmarks[234], root_user_landmarks[454], root_user_landmarks[14]]

print(root_user_landmarks[1])
# Release the FaceDetection module before entering the real-time loop
face_detection = None

# Initialize the tracker
tracker = cv2.TrackerCSRT_create()
tracking_initiated = False

# Set the frame width and height for performance improvement
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Define a threshold for landmark matching
landmark_threshold = 0.2

# Run real-time face detection with bounding box drawing and tracking
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Resize the frame for faster processing
    frame = cv2.resize(frame, (frame_width, frame_height))

    try:
        # Process the frame with face detection
        with mp.solutions.face_mesh.FaceMesh(refine_landmarks=True) as face_detection:
            results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            landmark_points = results.multi_face_landmarks
            current_face_landmarks = landmark_points[0].landmark

        # Draw bounding box around the detected face(s)
        if current_face_landmarks:
            current_face_landmarks_check = [current_face_landmarks[1], current_face_landmarks[468],
                                            current_face_landmarks[473], current_face_landmarks[234],
                                            current_face_landmarks[454], current_face_landmarks[14]]
            landmarks_distance = np.mean(np.sqrt(
                (current_face_landmarks_check[0].x - root_user_landmarks_check[0].x) ** 2 +
                (current_face_landmarks_check[0].y - root_user_landmarks_check[0].y) ** 2
            ))

            # If the landmarks are close, consider it a match
            if landmarks_distance < landmark_threshold:
                left = [current_face_landmarks[145], current_face_landmarks[159]]
                right = [current_face_landmarks[374], current_face_landmarks[386]]
                cv2.circle(frame, (int(current_face_landmarks[1].x * frame_width),
                                   int(current_face_landmarks[1].y * frame_height)), 3, (0, 255, 255))
                cv2.circle(frame, (int(current_face_landmarks[145].x * frame_width),
                                   int(current_face_landmarks[145].y * frame_height)), 3, (0, 255, 255))
                cv2.circle(frame, (int(current_face_landmarks[159].x * frame_width),
                                   int(current_face_landmarks[159].y * frame_height)), 3, (0, 255, 255))
                cv2.circle(frame, (int(current_face_landmarks[374].x * frame_width),
                                   int(current_face_landmarks[374].y * frame_height)), 3, (0, 255, 255))
                cv2.circle(frame, (int(current_face_landmarks[386].x * frame_width),
                                   int(current_face_landmarks[386].y * frame_height)), 3, (0, 255, 255))
                print("Nose\n", current_face_landmarks[1])
                print("Left Eye", current_face_landmarks[145], current_face_landmarks[159])
                print("Right Eye", current_face_landmarks[374], current_face_landmarks[386])
                threading.Thread(target=eye_motion, args=(left, right)).start()
                threading.Thread(target=cursor,args=(current_face_landmarks[1])).start()

    except Exception as e:
        import traceback
        traceback.print_exc()  # Print the entire exception traceback
        print(f"Error during face detection: {e}")
'''
'''
import cv2
import mediapipe as mp
import numpy as np
import threading
import pyautogui as py

frame_width = 640
frame_height = 480


def eye_motion(left_eye, right_eye):
    left = left_eye
    right = right_eye

    for landmark in left:
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)
        cv2.circle(frame, (x, y), 3, (0, 255, 255))

    for landmark in right:
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)
        cv2.circle(frame, (x, y), 3, (0, 255, 255))

    print("Right: ", (right[0].y - right[1].y))
    print("Left: ", (left[0].y - left[1].y))

    if (left[0].y - left[1].y) < 0.004 and (right[0].y - right[1].y) > 0.015:
        py.click(button='left')  # Perform left click

    if (right[0].y - right[1].y) < 0.004 and (left[0].y - left[1].y) > 0.015:
        py.click(button='right')  # Perform right click


def cursor(nose_landmark):
    x, y, z = [int(a * frame_width) for a in (nose_landmark.x, nose_landmark.y, nose_landmark.z)]
    current_x, current_y = py.position()
    py.moveTo(current_x + x, current_y + y)


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Initialize Mediapipe Face Detection module
face_detection = face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Check if the face_detection object is successfully initialized
if not face_detection:
    print("Error: Failed to initialize FaceDetection module.")
    exit()

# Capture the initial frame to get the root user's image
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        print("Failed to capture initial frame.")
        break

    cv2.imshow('Capture Root User Image', frame)

    # Press 'Space' key to capture the root user's image
    if cv2.waitKey(1) & 0xFF == ord(' '):
        root_user_image = frame.copy()
        break

cap.release()
cv2.destroyAllWindows()

# Process the root user image to get the facial landmarks
with mp.solutions.face_mesh.FaceMesh(refine_landmarks=True) as face_detection:
    root_user_results = face_detection.process(cv2.cvtColor(root_user_image, cv2.COLOR_BGR2RGB))
    landmark_points = root_user_results.multi_face_landmarks
    root_user_landmarks = landmark_points[0].landmark
    root_user_landmarks_check = [root_user_landmarks[1], root_user_landmarks[468], root_user_landmarks[473],
                                  root_user_landmarks[234], root_user_landmarks[454], root_user_landmarks[14]]

print(root_user_landmarks[1])
# Release the FaceDetection module before entering the real-time loop
face_detection = None

# Initialize the tracker
tracker = cv2.TrackerCSRT_create()
tracking_initiated = False

# Set the frame width and height for performance improvement
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Define a threshold for landmark matching
landmark_threshold = 0.2

# Run real-time face detection with bounding box drawing and tracking
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Resize the frame for faster processing
    frame = cv2.resize(frame, (frame_width, frame_height))

    try:
        # Process the frame with face detection
        with mp.solutions.face_mesh.FaceMesh(refine_landmarks=True) as face_detection:
            results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            landmark_points = results.multi_face_landmarks
            current_face_landmarks = landmark_points[0].landmark

        # Draw bounding box around the detected face(s)
        if current_face_landmarks:
            current_face_landmarks_check = [current_face_landmarks[1], current_face_landmarks[468],
                                            current_face_landmarks[473], current_face_landmarks[234],
                                            current_face_landmarks[454], current_face_landmarks[14]]
            landmarks_distance = np.mean(np.sqrt(
                (current_face_landmarks_check[0].x - root_user_landmarks_check[0].x) ** 2 +
                (current_face_landmarks_check[0].y - root_user_landmarks_check[0].y) ** 2
            ))

            # If the landmarks are close, consider it a match
            if landmarks_distance < landmark_threshold:
                left = [current_face_landmarks[145], current_face_landmarks[159]]
                right = [current_face_landmarks[374], current_face_landmarks[386]]
                cv2.circle(frame, (int(current_face_landmarks[1].x * frame_width),
                                   int(current_face_landmarks[1].y * frame_height)), 3, (0, 255, 255))
                cv2.circle(frame, (int(current_face_landmarks[145].x * frame_width),
                                   int(current_face_landmarks[145].y * frame_height)), 3, (0, 255, 255))
                cv2.circle(frame, (int(current_face_landmarks[159].x * frame_width),
                                   int(current_face_landmarks[159].y * frame_height)), 3, (0, 255, 255))
                cv2.circle(frame, (int(current_face_landmarks[374].x * frame_width),
                                   int(current_face_landmarks[374].y * frame_height)), 3, (0, 255, 255))
                cv2.circle(frame, (int(current_face_landmarks[386].x * frame_width),
                                   int(current_face_landmarks[386].y * frame_height)), 3, (0, 255, 255))
                print("Nose\n", current_face_landmarks[1])
                print("Left Eye", current_face_landmarks[145], current_face_landmarks[159])
                print("Right Eye", current_face_landmarks[374], current_face_landmarks[386])
                threading.Thread(target=eye_motion, args=(left, right)).start()
                threading.Thread(target=cursor, args=(current_face_landmarks[1])).start()

    except Exception as e:
        import traceback
        traceback.print_exc()  # Print the entire exception traceback
        print(f"Error during face detection: {e}")
'''
'''
import cv2
import mediapipe as mp
import numpy as np
import threading
import pyautogui as py

frame_width = 640
frame_height = 480

# Initialize the previous cursor position
prev_cursor_position = (0, 0)

def eye_motion(left_eye, right_eye):
    left = left_eye
    right = right_eye

    for landmark in left:
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)
        cv2.circle(frame, (x, y), 3, (0, 255, 255))

    for landmark in right:
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)
        cv2.circle(frame, (x, y), 3, (0, 255, 255))

    print("Right: ", (right[0].y - right[1].y))
    print("Left: ", (left[0].y - left[1].y))

    if (left[0].y - left[1].y) < 0.004 and (right[0].y - right[1].y) > 0.015:
        py.click(button='left')  # Perform left click

    if (right[0].y - right[1].y) < 0.004 and (left[0].y - left[1].y) > 0.015:
        py.click(button='right')  # Perform right click

def cursor(nose_landmark):
    global prev_cursor_position
    x, y, z = [int(a * frame_width) for a in (nose_landmark.x, nose_landmark.y, nose_landmark.z)]

    # Calculate the difference between current and previous cursor positions
    delta_x = x - prev_cursor_position[0]
    delta_y = y - prev_cursor_position[1]

    # Move the cursor relative to its current position
    current_x, current_y = py.position()
    py.moveTo(current_x + delta_x, current_y + delta_y)

    # Update the previous cursor position
    prev_cursor_position = (x, y)

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Initialize Mediapipe Face Detection module
face_detection = face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Check if the face_detection object is successfully initialized
if not face_detection:
    print("Error: Failed to initialize FaceDetection module.")
    exit()

# Capture the initial frame to get the root user's image
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        print("Failed to capture initial frame.")
        break

    cv2.imshow('Capture Root User Image', frame)

    # Press 'Space' key to capture the root user's image
    if cv2.waitKey(1) & 0xFF == ord(' '):
        root_user_image = frame.copy()
        break

cap.release()
cv2.destroyAllWindows()

# Process the root user image to get the facial landmarks
with mp.solutions.face_mesh.FaceMesh(refine_landmarks=True) as face_detection:
    root_user_results = face_detection.process(cv2.cvtColor(root_user_image, cv2.COLOR_BGR2RGB))
    landmark_points = root_user_results.multi_face_landmarks
    root_user_landmarks = landmark_points[0].landmark
    root_user_landmarks_check = [root_user_landmarks[1], root_user_landmarks[468], root_user_landmarks[473],
                                  root_user_landmarks[234], root_user_landmarks[454], root_user_landmarks[14]]

print(root_user_landmarks[1])
# Release the FaceDetection module before entering the real-time loop
face_detection = None

# Initialize the tracker
tracker = cv2.TrackerCSRT_create()
tracking_initiated = False

# Set the frame width and height for performance improvement
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Define a threshold for landmark matching
landmark_threshold = 0.2

# Run real-time face detection with bounding box drawing and tracking
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        print("Failed to capture frame.")
        break

    # Resize the frame for faster processing
    frame = cv2.resize(frame, (frame_width, frame_height))

    try:
        # Process the frame with face detection
        with mp.solutions.face_mesh.FaceMesh(refine_landmarks=True) as face_detection:
            results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            landmark_points = results.multi_face_landmarks
            current_face_landmarks = landmark_points[0].landmark

        # Draw bounding box around the detected face(s)
        if current_face_landmarks:
            current_face_landmarks_check = [current_face_landmarks[1], current_face_landmarks[468],
                                            current_face_landmarks[473], current_face_landmarks[234],
                                            current_face_landmarks[454], current_face_landmarks[14]]
            landmarks_distance = np.mean(np.sqrt(
                (current_face_landmarks_check[0].x - root_user_landmarks_check[0].x) ** 2 +
                (current_face_landmarks_check[0].y - root_user_landmarks_check[0].y) ** 2
            ))

            # If the landmarks are close, consider it a match
            if landmarks_distance < landmark_threshold:
                left = [current_face_landmarks[145], current_face_landmarks[159]]
                right = [current_face_landmarks[374], current_face_landmarks[386]]
                cv2.circle(frame, (int(current_face_landmarks[1].x * frame_width),
                                   int(current_face_landmarks[1].y * frame_height)), 3, (0, 255, 255))
                cv2.circle(frame, (int(current_face_landmarks[145].x * frame_width),
                                   int(current_face_landmarks[145].y * frame_height)), 3, (0, 255, 255))
                cv2.circle(frame, (int(current_face_landmarks[159].x * frame_width),
                                   int(current_face_landmarks[159].y * frame_height)), 3, (0, 255, 255))
                cv2.circle(frame, (int(current_face_landmarks[374].x * frame_width),
                                   int(current_face_landmarks[374].y * frame_height)), 3, (0, 255, 255))
                cv2.circle(frame, (int(current_face_landmarks[386].x * frame_width),
                                   int(current_face_landmarks[386].y * frame_height)), 3, (0, 255, 255))
                print("Nose\n", current_face_landmarks[1])
                print("Left Eye", current_face_landmarks[145], current_face_landmarks[159])
                print("Right Eye", current_face_landmarks[374], current_face_landmarks[386])
                threading.Thread(target=eye_motion, args=(left, right)).start()
                threading.Thread(target=cursor, args=(current_face_landmarks[1],)).start()

    except Exception as e:
        import traceback
        traceback.print_exc()  # Print the entire exception traceback
        print(f"Error during face detection: {e}")
'''

import cv2
import mediapipe as mp
import numpy as np
import threading
import pyautogui as py

frame_width = 640
frame_height = 480

# Initialize the previous cursor position
prev_cursor_position = (0, 0)

def eye_motion(left_eye, right_eye):
    left = left_eye
    right = right_eye

    for landmark in left:
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)
        cv2.circle(frame, (x, y), 3, (0, 255, 255))

    for landmark in right:
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)
        cv2.circle(frame, (x, y), 3, (0, 255, 255))

    print("Right: ", (right[0].y - right[1].y))
    print("Left: ", (left[0].y - left[1].y))

    if (left[0].y - left[1].y) < 0.004 and (right[0].y - right[1].y) > 0.015:
        py.click(button='left')  # Perform left click

    if (right[0].y - right[1].y) < 0.004 and (left[0].y - left[1].y) > 0.015:
        py.click(button='right')  # Perform right click

def cursor(nose_landmark):
    global prev_cursor_position
    x, y, z = [int(a * frame_width) for a in (nose_landmark.x, nose_landmark.y, nose_landmark.z)]

    # Calculate the difference between current and previous cursor positions
    delta_x = x - prev_cursor_position[0]
    delta_y = y - prev_cursor_position[1]

    # Move the cursor relative to its current position
    current_x, current_y = py.position()
    py.moveTo(current_x + 5*delta_x, current_y + 5*delta_y)

    # Update the previous cursor position
    prev_cursor_position = (x, y)

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Initialize Mediapipe Face Detection module
face_detection = face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Check if the face_detection object is successfully initialized
if not face_detection:
    print("Error: Failed to initialize FaceDetection module.")
    exit()

# Capture the initial frame to get the root user's image
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        print("Failed to capture initial frame.")
        break

    cv2.imshow('Capture Root User Image', frame)

    # Press 'Space' key to capture the root user's image
    if cv2.waitKey(1) & 0xFF == ord(' '):
        root_user_image = frame.copy()
        break

cap.release()
cv2.destroyAllWindows()

# Process the root user image to get the facial landmarks
with mp.solutions.face_mesh.FaceMesh(refine_landmarks=True) as face_detection:
    root_user_results = face_detection.process(cv2.cvtColor(root_user_image, cv2.COLOR_BGR2RGB))
    landmark_points = root_user_results.multi_face_landmarks
    root_user_landmarks = landmark_points[0].landmark
    root_user_landmarks_check = [root_user_landmarks[1], root_user_landmarks[468], root_user_landmarks[473],
                                  root_user_landmarks[234], root_user_landmarks[454], root_user_landmarks[14]]

print(root_user_landmarks[1])
# Release the FaceDetection module before entering the real-time loop
face_detection = None

# Initialize the tracker
tracker = cv2.TrackerCSRT_create()
tracking_initiated = False

# Set the frame width and height for performance improvement
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Define a threshold for landmark matching
landmark_threshold = 0.2

# Run real-time face detection with bounding box drawing and tracking
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        print("Failed to capture frame.")
        break

    # Resize the frame for faster processing
    frame = cv2.resize(frame, (frame_width, frame_height))

    try:
        # Process the frame with face detection
        with mp.solutions.face_mesh.FaceMesh(refine_landmarks=True) as face_detection:
            results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            landmark_points = results.multi_face_landmarks
            current_face_landmarks = landmark_points[0].landmark

        # Draw bounding box around the detected face(s)
        if current_face_landmarks:
            current_face_landmarks_check = [current_face_landmarks[1], current_face_landmarks[468],
                                            current_face_landmarks[473], current_face_landmarks[234],
                                            current_face_landmarks[454], current_face_landmarks[14]]
            landmarks_distance = np.mean(np.sqrt(
                (current_face_landmarks_check[0].x - root_user_landmarks_check[0].x) ** 2 +
                (current_face_landmarks_check[0].y - root_user_landmarks_check[0].y) ** 2
            ))

            # If the landmarks are close, consider it a match
            #if landmarks_distance < landmark_threshold:
            left = [current_face_landmarks[145], current_face_landmarks[159]]
            right = [current_face_landmarks[374], current_face_landmarks[386]]
            cv2.circle(frame, (int(current_face_landmarks[1].x * frame_width),
                                   int(current_face_landmarks[1].y * frame_height)), 3, (0, 255, 255))
            cv2.circle(frame, (int(current_face_landmarks[145].x * frame_width),
                                   int(current_face_landmarks[145].y * frame_height)), 3, (0, 255, 255))
            cv2.circle(frame, (int(current_face_landmarks[159].x * frame_width),
                                   int(current_face_landmarks[159].y * frame_height)), 3, (0, 255, 255))
            cv2.circle(frame, (int(current_face_landmarks[374].x * frame_width),
                                   int(current_face_landmarks[374].y * frame_height)), 3, (0, 255, 255))
            cv2.circle(frame, (int(current_face_landmarks[386].x * frame_width),
                                   int(current_face_landmarks[386].y * frame_height)), 3, (0, 255, 255))
            print("Nose\n", current_face_landmarks[1])
            print("Left Eye", current_face_landmarks[145], current_face_landmarks[159])
            print("Right Eye", current_face_landmarks[374], current_face_landmarks[386])
            threading.Thread(target=eye_motion, args=(left, right)).start()
            threading.Thread(target=cursor, args=(current_face_landmarks[1],)).start()
                # eye_motion(left,right)
                # cursor(current_face_landmarks[1])
    except Exception as e:
        import traceback
        traceback.print_exc()  # Print the entire exception traceback
        print(f"Error during face detection: {e}")
        
    cv2.imshow("real",frame)
    
    if cv2.waitKey(1) &0xFF==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
