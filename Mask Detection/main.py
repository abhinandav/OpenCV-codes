import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained face detection model from OpenCV
face_net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000_fp16.caffemodel')

# Load the mask detection model (assumed pre-trained and saved as mask_detector.model)
mask_net = load_model('mask_detector.model')

# Function to detect face mask status
def detect_and_predict_mask(frame, face_net, mask_net):
    # Get frame dimensions
    (h, w) = frame.shape[:2]
    
    # Preprocess the frame for face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    
    faces = []
    locs = []
    preds = []
    
    # Loop over detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Ensure box is within frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            # Extract face ROI
            face = frame[startY:endY, startX:endX]
            face = cv2.resize(face, (224, 224))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = np.array(face) / 255.0
            face = np.expand_dims(face, axis=0)
            
            # Save face and its bounding box
            faces.append(face)
            locs.append((startX, startY, endX, endY))
    
    # Predict mask status if at least one face was detected
    if len(faces) > 0:
        preds = mask_net.predict(faces)
    
    return (locs, preds)

# Video stream from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    # Detect faces and predict mask status
    (locs, preds) = detect_and_predict_mask(frame, face_net, mask_net)
    
    # Loop over faces detected and associated mask predictions
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        
        # Determine label and color
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        
        # Display label and bounding box
        cv2.putText(frame, f"{label}: {max(mask, withoutMask) * 100:.2f}%", (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    
    # Show the output
    cv2.imshow("Frame", frame)
    
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
