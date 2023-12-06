import cv2
import os
import time
import shutil

#  placeholder function
def detect(face_image):
    return f"{int(time.time())}"





photos_folder = 'temp_photos'
if os.path.exists(photos_folder):
    shutil.rmtree(photos_folder)
os.mkdir(photos_folder)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

counter = 0
time_var = 125   # increase this to increase time between shots
detected_text = ""
while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Increase the bounding box size by multiplying w and h with a scaling factor
        padding = 0.3  # Change this value to adjust the size of the bounding box
        
        # Calculate new x, y, w, h with increased size
        new_w = int(w * (1 + padding))
        new_h = int(h * (1 + padding))
        new_x = max(int(x - (padding * w / 2)), 0)
        new_y = max(int(y - (padding * h / 2)), 0)
        
        # bounding box
        cv2.rectangle(frame, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 255, 0), 2)
        
        # Crop the frame according to the adjusted coordinates
        # cropped_face = frame[new_y:new_y+new_h, new_x:new_x+new_w]
        # detected_text = detect(cropped_face)
        cv2.putText(frame, detected_text, (new_x, new_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save the cropped face after a time interval
        if counter % time_var == 0:
            cropped_face = frame[new_y:new_y+new_h, new_x:new_x+new_w]
            detected_text = detect(cropped_face)
            # cv2.putText(frame, detected_text, (new_x, new_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
            file_name = f'{photos_folder}/face_{int(time.time())}.jpg'
            cv2.imwrite(file_name, cropped_face)
    
    counter += 1
        
    cv2.imshow('Face Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    
cap.release()
cv2.destroyAllWindows()