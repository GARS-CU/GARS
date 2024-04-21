import cv2
import numpy as np
import contextlib
import sys
import os
import pickle
from PIL import Image
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Classes'))
from Daisee_Regressor_Final import EngagementClassifierInference
# from deepface import DeepFace
sys.path.append(os.environ['GARS_PROJ'])  #append path for util 
sys.path.append(os.path.join(os.environ['GARS_PROJ'], 'art_generate')) #append path for rec system 
from util import *
from art_rec_bog import ArtRecSystem

class Engagement:
    def __init__(self):
        # Initialize the Haar Cascade Classifier
        haar_path =  os.path.join("Models", "haarcascade_frontalface_alt.xml")
        self.face_cascade = cv2.CascadeClassifier(haar_path)
        self.rec = ArtRecSystem(metric='cosine', art_generate=True)
    
    def generate_art(self, engagement_score):
        image, prompt, words = self.rec(rating=engagement_score)

        # Save gen image
        generated_images_dir = 'Generated_Images'
        os.makedirs(generated_images_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(generated_images_dir, f"generated_art_{timestamp}.png")
        
        image.save(image_path)
        print(prompt)
        print(words)

        return image_path
        
    def extract_open_features(self, filename):
        openface_path = "docker exec openface_docker /home/openface-build/build/bin/"
        video_path = os.path.join("received_videos", filename)
        output_dir = "processed" 

        # Execute OpenFace FeatureExtraction
        command = f'{openface_path}FeatureExtraction -f {video_path} -out_dir {output_dir}'
        os.system(command)

        open_csv_dir = os.path.join('/zooper2/colin.hwang/openface_dump/processed', f"{os.path.splitext(filename)[0]}.csv")
        data = np.loadtxt(open_csv_dir, delimiter=",", skiprows=1)
        
        # Ensure only the first 300 frames are considered
        if data.shape[0] > 300:
            data = data[:300, :]

        # The first 5 columns are not features
        open_features = data[:, 5:]
        
        # Check if the shape matches the expected (300, 709)
        if open_features.shape != (300, 709):
            raise ValueError("Unexpected number of features. Expected 709 features per frame.")

        return open_features
    
    def extract_emotion_features(self, video_path):
        # Get frame count and capture a frame per second
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []
        current_frame = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Doing cap.set kind of messes things up for some reason, capturing only 4 frames
            # Skipping 30 frames per cap.read() I think is inaccurate, so instead read frame by frame
            # Doing this works, making sure that we're capturing a frame every 30 frames
            if current_frame % fps == 0 and len(frames) < 10:
                # print(current_frame)
                # Gray scale aand detect face
                processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_locations = self.face_cascade.detectMultiScale(processed_frame, scaleFactor=1.05, minNeighbors=5)
                
                # If no face is detected, use 0 array, otherwise use largest face found
                if len(face_locations) == 0:
                    processed_frame = np.zeros((48, 48))
                else:
                    x, y, w, h = max(face_locations, key=lambda x: x[2] * x[3])
                    processed_frame = processed_frame[y:y+h, x:x+w]
                
                processed_frame = cv2.resize(processed_frame, (48, 48))
                frames.append(processed_frame)
                
                # Break once we have 10 frames. Videos may be longer than 10 seconds
                if len(frames) == 10:
                    break 

            current_frame += 1

        cap.release()
        frames_array = np.array(frames).reshape(-1, 48, 48, 1).astype("float32")
        
        return frames_array
    
# def main():
#     engagement = Engagement()
    
#     video_file = 'uploaded_video.mp4'
#     csv_file = 'uploaded_video.csv'
    
#     open_features = engagement.extract_open_features(csv_file)
#     print("Open Features Shape:", open_features.shape)
#     # print(open_features)
    
#     emotion_features = engagement.extract_emotion_features(video_file)
#     print("Emotion Features Shape:", emotion_features.shape)
#     # print(emotion_features)

#     classifier = EngagementClassifierInference()

#     open_features_pca = classifier.get_open_inference(open_features)
#     print("Open Features PCA Shape:", open_features.shape)

#     engagement_score = classifier.predict_engagement(emotion_features, open_features_pca)
#     print(engagement_score)

if __name__ == "__main__":
    main()
