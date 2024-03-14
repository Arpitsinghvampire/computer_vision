import cv2
import mediapipe as mp
import pyautogui as py
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = py.size()
while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape
    if landmark_points:
        landmarks = landmark_points[0].landmark
        # for id, landmark in enumerate(landmarks[474:478]):
        #     x = int(landmark.x * frame_w)
        #     y = int(landmark.y * frame_h)
        #     cv2.circle(frame, (x, y), 3, (0, 255, 0)) 
        #     if id == 1:
        #         screen_x = screen_w* landmark.x
        #         screen_y = screen_h* landmark.y
        #         py.move(screen_x,screen_y)
        left = [landmarks[145], landmarks[159]]
        right=[landmarks[374], landmarks[386]]
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))
        
        for landmark in right:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))
        print("Right: ", (right[0].y - right[1].y))
        print("Left: ", (left[0].y - left[1].y))
        if (left[0].y - left[1].y) < 0.004 and (right[0].y - right[1].y) > 0.015:
            py.leftClick()#500,850,1,1,
            
        if (right[0].y - right[1].y) < 0.004 and (left[0].y - left[1].y) > 0.015:
            py.rightClick()#500,850,1,1,   
            # py.hotkey('winkey','H')
            
    cv2.imshow('Eye Controlled Mouse', frame)
    if cv2.waitKey(1) &0xFF==ord('q'):
        break
    cv2.waitKey(1)
