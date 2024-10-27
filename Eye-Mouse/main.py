import cv2
import mediapipe as mp
import pyautogui

# Initialize FaceMesh with refine_landmarks=True
face_mesh_landmarks = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Capture video from the webcam
width,height=1280,720
webcam=cv2.VideoCapture(0)
webcam.set(3,width)
webcam.set(4,height)
screen_w, screen_h = pyautogui.size()

while True:
    _, img = webcam.read()
    img = cv2.flip(img, 1)
    window_h, window_w, _ = img.shape  

    # Convert the image to RGB for mediapipe processing
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    processed_image = face_mesh_landmarks.process(rgb_img)
    all_face_landmarks = processed_image.multi_face_landmarks

    if all_face_landmarks:
        # Get the landmarks of the first detected face
        one_face_landmark_points = all_face_landmarks[0].landmark

        # Iterate over specific landmark points related to the eye region
        for id, landmark_point in enumerate(one_face_landmark_points[474:478]):
            x = int(landmark_point.x * window_w)
            y = int(landmark_point.y * window_h)

            # Move cursor based on the specific eye landmark
            if id == 1:
                mouse_x = int(landmark_point.x * screen_w)
                mouse_y = int(landmark_point.y * screen_h)
                pyautogui.moveTo(mouse_x, mouse_y)

            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

        # Get left eye landmark points for blink detection
        left_eye = [one_face_landmark_points[145], one_face_landmark_points[159]]
        for landmark_point in left_eye:
            x = int(landmark_point.x * window_w)
            y = int(landmark_point.y * window_h)
            cv2.circle(img, (x, y), 3, (0, 255, 255), -1)

        # Check for blink by measuring the vertical distance between the two eye points
        if (left_eye[0].y - left_eye[1].y) < 0.01:
            pyautogui.click()
            pyautogui.sleep(2)
            print('mouse clicked')

    # Show the video with eye-tracking
    cv2.imshow("Eye-controlled mouse", img)
    
    # Break the loop if the ESC key is pressed
    key = cv2.waitKey(10)
    if key == 27:
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()
