import cv2 as cv
import mediapipe as mp
import time
ctime=0
ptime=0
mpPose=mp.solutions.pose
pose=mpPose.Pose()
mpDraw=mp.solutions.drawing_utils
cap=cv.VideoCapture(0)
while True:
    success,image=cap.read()
    imagrgb=cv.cvtColor(image,cv.COLOR_BGR2RGB)
    results=pose.process(imagrgb)
    print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(image,results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id,lm in enumerate(results.pose_landmarks.landmark):
            print(id,lm)
            h,w,c=image.shape
            cx,cy=int(lm.x*w),int(lm.y*h)
            cv.circle(image,(cx,cy),10,(255,0,0),cv.FILLED)




    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime
    cv.putText(image,str(int(fps)),(70,50),cv.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
    cv.imshow("IMAGE",image)
    cv.waitKey(1)
