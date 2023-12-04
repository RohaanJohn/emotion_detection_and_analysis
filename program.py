import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret: break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    cv2.imshow('Face Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    
cap.release()
cv2.destroyAllWindows()










# import cv2

# # Load the pre-trained Haar Cascade face classifier
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# # Read the input image
# image = cv2.imread('noel.jpg')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Detect faces
# faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# # Draw rectangles around detected faces
# for (x, y, w, h) in faces:
#     cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# # Display the result
# cv2.imshow('Detected Faces', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()