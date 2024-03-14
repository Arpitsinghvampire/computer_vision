import cv2
import numpy as np

cap=cv2.VideoCapture(0) #this takes in 0 as the main web camera
while True:
    suc,frame=cap.read()
    
    frame=cv2.flip(frame,1)  
    laplacian=cv2.Laplacian(frame,cv2.CV_64F) 
    laplacian=np.uint8(laplacian)
    
    k=np.hstack([frame,laplacian])
    cv2.imshow("LAPLACIAN",laplacian)
    
    if cv2.waitKey(1) &0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
