import cv2
import os
import time
import shutil
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

#  placeholder function
def detect(file_name):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model("D:/projects/Dhwani/emotion_detection_and_analysis/keras_Model.h5", compile=False)

    # Load the labels
    class_names = open("D:/projects/Dhwani/emotion_detection_and_analysis/labels.txt", "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(file_name).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return(class_name[2:])

    

photos_folder = 'temp_photos'
if os.path.exists(photos_folder):
    shutil.rmtree(photos_folder)
os.mkdir(photos_folder)

face_cascade = cv2.CascadeClassifier('D:/projects/Dhwani/emotion_detection_and_analysis/haarcascade_frontalface_default.xml')

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
            #detected_text = detect(cropped_face)
            # cv2.putText(frame, detected_text, (new_x, new_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
            file_name = f'{photos_folder}/face_{int(time.time())}.jpg'
            cv2.imwrite(file_name, cropped_face)
            detected_text = detect(file_name)
    
    counter += 1
        
    cv2.imshow('Face Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    
cap.release()
cv2.destroyAllWindows()