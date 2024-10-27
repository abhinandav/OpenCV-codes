import cv2
from cvzone.HandTrackingModule import HandDetector
import socket

# Parameters
width,height=1280,720

# Webcam
cap=cv2.VideoCapture(0)
cap.set(3,width)
cap.set(4,height)

# Hand Detector
detector=HandDetector(maxHands=2,detectionCon=0.8)

# Communication

sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
serverAddressPort=("127.0.0.1",5052)




while True:
    # Get the frame from the webcam
    success,img=cap.read()
    img = cv2.flip(img, 1)
    # Hands
    hands,img=detector.findHands(img)

    # Land mark values -(x-y-z)*21
    data=[]
    if hands:
        #Get the first hand detected
        hand=hands[0]
        # Get the landmark list
        lmList=hand['lmList']
        for lm in lmList:
            data.extend([lm[0],height-lm[1],lm[2]])
        sock.sendto(str.encode(str(data)),serverAddressPort)


    cv2.imshow("Image",img)
    key = cv2.waitKey(10)
    if key == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()