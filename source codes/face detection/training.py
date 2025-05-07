import os
import pickle
import face_recognition

# Path to the dataset folder
dataset_path = 'dataset'
# Path to store the trained model (encodings and names)
trained_model_path = 'faces.pkl'

# Function to train the model from the dataset
def train_model(dataset_path, trained_model_path):
    known_face_encodings = []
    known_face_names = []

    # Loop through the dataset folder and encode faces
    for filename in os.listdir(dataset_path):
        if filename.endswith('.jpg'):  # Ensure we're only processing .jpg files
            # Load the image from the dataset folder
            image = face_recognition.load_image_file(f"{dataset_path}/{filename}")
            
            # Get the face encodings for the image (there should be only one face per image)
            face_encoding = face_recognition.face_encodings(image)
            
            if face_encoding:  # Ensure that the face encoding was successfully extracted
                known_face_encodings.append(face_encoding[0])  # The first encoding is the face encoding
                known_face_names.append(filename.split('.')[0])  # Name is extracted from the filename
                print(f"Successfully trained {filename.split('.')[0]}")  # Print the name of the person trained
            else:
                print(f"Warning: No face found in {filename}, skipping.")

    # Save the trained model to a pickle file
    with open(trained_model_path, 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)

    print(f"Model trained and saved to {trained_model_path}")

# Train the model
train_model(dataset_path, trained_model_path)
