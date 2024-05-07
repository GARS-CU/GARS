import cv2
import numpy as np
import contextlib
import sys
import os
import pickle
from scipy.interpolate import interp1d
from PIL import Image
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Classes'))
from Daisee_Regressor_Final import EngagementClassifierInference
# from deepface import DeepFace
sys.path.append(os.environ['GARS_PROJ'])  #append path for util 
sys.path.append(os.path.join(os.environ['GARS_PROJ'], 'art_generate')) #append path for rec system 
#sys.path.append(os.path.join(os.environ['GARS_PROJ'], 'engagement', 'focus'))
from util import *


class Engagement:
    def __init__(self):
        # Initialize the Haar Cascade Classifier
        haar_path =  os.path.join("Models", "haarcascade_frontalface_alt.xml")
        self.face_cascade = cv2.CascadeClassifier(haar_path)
        # self.rec = ArtRecSystem(metric='cosine', art_generate=True)
    def generate_art(self, rec, engagement_score, iterations, total_iterations, session_path):
        if iterations > total_iterations:
            _, _, _ = rec(rating=engagement_score)
            return
        
        image, prompt, words = rec(rating=engagement_score)

        # Save gen image
        generated_images_dir = 'Generated_Images'
        os.makedirs(generated_images_dir, exist_ok=True)

        image_filename = f'art_{iterations:03}.png'
        image_path = os.path.join(session_path, 'art', image_filename)
            
        image.save(image_path)
        print(prompt)
        print(words)

        return image_path, iterations

    def extract_open_features(self, filename):
        # Run openface executable for feature extraction
        openface_path = "docker exec openface_docker /home/openface-build/build/bin/"
        video_path = os.path.join("received_videos", filename)
        output_dir = "processed"

        command = f'{openface_path}FeatureExtraction -f {video_path} -out_dir {output_dir}'
        os.system(command)

        open_csv_dir = os.path.join(f"/zooper2/{os.getenv('USER')}/openface_dump/processed", f"{os.path.splitext(filename)[0]}.csv")
        data = np.loadtxt(open_csv_dir, delimiter=",", skiprows=1)

        feature_data = data[:, 5:]

        # Normalize the frame data to exactly 300 frames
        current_frame_count = feature_data.shape[0]
        feature_count = feature_data.shape[1]
        target_frame_count = 300

        # Create an index array for current data
        current_index = np.linspace(0, current_frame_count - 1, num=current_frame_count)
        # Create a target index array for 300 frames
        target_index = np.linspace(0, current_frame_count - 1, num=target_frame_count)

        # Interpolation function for each feature column
        interpolated_data = np.zeros((target_frame_count, feature_count))
        for i in range(feature_count):
            interpolator = interp1d(current_index, feature_data[:, i], kind='linear')
            interpolated_data[:, i] = interpolator(target_index)

        # Check if the shape matches the expected (300, feature_count)
        if interpolated_data.shape != (300, feature_count):
            raise ValueError(f"Unexpected number of features. Expected {feature_count} features per frame.")

        return interpolated_data, current_frame_count

    
    def extract_emotion_features(self, video_path, total_frames):
        cap = cv2.VideoCapture(video_path)
        if total_frames < 10:
            interval = 1
        else:
            interval = total_frames // 10

        frames = []
        selected_frames = set()

        for i in range(10):
            frame_index = min(i * interval, total_frames - 1)
            selected_frames.add(frame_index)

        current_frame = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame in selected_frames:
                processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_locations = self.face_cascade.detectMultiScale(processed_frame, scaleFactor=1.05, minNeighbors=5)

                if len(face_locations) == 0:
                    processed_frame = np.zeros((48, 48))
                else:
                    x, y, w, h = max(face_locations, key=lambda x: x[2] * x[3])
                    processed_frame = processed_frame[y:y+h, x:x+w]

                processed_frame = cv2.resize(processed_frame, (48, 48))
                frames.append(processed_frame)
                selected_frames.remove(current_frame)

                if not selected_frames:
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

