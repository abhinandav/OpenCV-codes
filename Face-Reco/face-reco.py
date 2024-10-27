# Detecting faces in the camera

import cv2

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

webcam=cv2.VideoCapture(0) # 0 bcz we have only one webcam
while True:
    _,img=webcam.read()
    # convert image to b&w
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # detect multiple faces(4)
    faces= face_cascade.detectMultiScale(gray,1.5,4)
    # 
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
    # show cam
    cv2.imshow("Face detection",img)
    # wait for 10 milli second
    key=cv2.waitKey(10)
    if key==27:
        break

# close all
webcam.release()
cv2.destroyAllWindows()

