import cv2
import os

# Get the name of the person (this will be used as part of the image filename)
person_name = input("Enter the name of the person: ")

# Create a folder to save captured images for the specific person
dataset_path = 'dataset'
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# To capture 50 images of the face for training
count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to grayscale (for better performance)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load the face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Save the captured face images with the person's name and count
        count += 1
        cv2.imwrite(f"{dataset_path}/{person_name}.{count}.jpg", gray[y:y + h, x:x + w])

    # Display the video feed
    cv2.imshow("Capturing Face", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Stop after capturing 50 images
    if count >= 50:
        break

cap.release()
cv2.destroyAllWindows()

print(f"Captured {count} images for {person_name}.")
