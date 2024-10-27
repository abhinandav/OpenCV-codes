import cv2
import mediapipe
import pyautogui

capture_hands=mediapipe.solutions.hands.Hands()
drawing_option=mediapipe.solutions.drawing_utils

camera=cv2.VideoCapture(0)

while True:
    _,img=camera.read()
    image_h,image_w,_=img.shape
    img=cv2.flip(img,1)


    rgb_image=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    output_hands=capture_hands.process(rgb_image)
    all_hands=output_hands.multi_hand_landmarks
    if all_hands:
        for hand in all_hands:
            drawing_option.draw_landmarks(img,hand)
            one_hand_landmarks=hand.landmark
            for id,lm in enumerate(one_hand_landmarks):
                x=lm.x




    cv2.imshow('Hand Mouse',img)

    key=cv2.waitKey(100)
    if key==27:
        break

camera.release()
cv2.destroyAllWindows()