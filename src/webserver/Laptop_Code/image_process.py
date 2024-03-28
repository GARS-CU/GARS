import cv2
import io
import os
import pickle
import time
import numpy as np

class Image:
    def __init__(self, image_data):
        # Convert the input image data to a numpy array and decode img
        nparr = np.frombuffer(image_data, np.uint8)
        self.image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def preprocess(self):
        # Convert to grayscale
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Load the Haar Cascade Classifier for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            # Extract the first face
            (x, y, w, h) = faces[0]
            # Crop and resize the image to center on the face
            face_centered = gray_image[y:y+h, x:x+w]
            face_resized = cv2.resize(face_centered, (48, 48))

            # Save preprocessed image for testing
            timestamp = int(time.time())
            save_dir = 'assets'
            os.makedirs(save_dir, exist_ok=True)

            # Save the preprocessed image
            filename = f"face_{timestamp}.png"
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, face_resized)

            # Pickle the image data
            with open(filepath, 'rb') as f:
                img_data = f.read()

            pickled_data = pickle.dumps(img_data)

            # Return the pickled data
            return pickled_data
        else:
            return None
