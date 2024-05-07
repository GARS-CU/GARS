from flask import Flask, make_response, request, jsonify, send_file
import os
from datetime import datetime
from engagement_preprocess import Engagement
import requests
import sys
import io
import pickle
import numpy as np
from PIL import Image
import csv
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Classes'))
from Daisee_Regressor_Final import EngagementClassifierInference
from art_rec_bog import ArtRecSystem
#from focus_calibrator import *
from gaze_wrapper import GazeWrapper
app = Flask(__name__)

@app.route('/process_data', methods=['POST'])
def process_data():
    global session_path, iterations, engagement_scores, gaze_wrapper, average_gaze, gaze_stddev

    print("Flask iteration: ", iterations)
    print("Gaze Count: ", len(gaze_scores))
        
    # Error checking
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400

    # Save video within session folder
    video_filename = f'video_{iterations:03}.mp4'
    video_filepath = os.path.join(session_path, 'videos', video_filename)

    file = request.files['file']
    file.save(video_filepath)
    average_duration = float(request.form.get('average_duration', 0))
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save another copy of the video for openface processing
    file.seek(0)
    videos_dir = f"/zooper2/{os.getenv('USER')}/openface_dump/received_videos"
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f'video_{datetime_str}.mp4'
    output_filepath = os.path.join(videos_dir, output_filename)
    file.save(output_filepath)
    
    try:
        openface_features, frame_count = engagement.extract_open_features(output_filename)
        if gaze_wrapper._gaze_model:
            gaze_scores_output = gaze_wrapper(output_filename)
            gaze_score = np.mean(gaze_scores_output)
            print("gaze scores", gaze_scores_output)
            
        emotions_features = engagement.extract_emotion_features(output_filepath, frame_count)
        open_features_pca = classifier.get_open_inference(openface_features)
        engagement_score = classifier.predict_engagement(emotions_features, open_features_pca)

        engagement_score = 2 * engagement_score - 1
        engagement_score = -1 * engagement_score

    except Exception as e:
        print(f"Error during processing: {e}")
        engagement_score = 0
        gaze_score = 0
    
    # Append scores
    engagement_scores.append((iterations, engagement_score))
    gaze_scores.append(gaze_score)

    # Increment counters
    iterations += 1
    
    # Calculate stddev of first 5 gaze scores
    if len(gaze_scores) >= 5:
        average_gaze = np.mean(gaze_scores)
        gaze_stddev = np.std(gaze_scores)

    print("Avg Duration: ", average_duration)
    if average_duration > 0 and len(gaze_scores) >= 5:
        # Add time to engagement score
        current_duration = float(request.form.get('duration', 0))
        std_dev = float(request.form.get('std_deviation', 0))
        z_score = (current_duration - average_duration) / std_dev
        adjustment_factor = z_score * 0.3
        print("Time StdDev: ", std_dev)
        print("Time z score: ", z_score)
        
        # Add gaze to score
        z_score_gaze = (gaze_score - average_gaze) / gaze_stddev
        adjustment_factor_gaze = z_score * 0.15
        print("Gaze StdDev: ", gaze_stddev)
        print("Gaze z score: ", z_score_gaze)        
        print("Average Gaze: ", average_gaze)
        print("Current gaze: ", gaze_score)    
        engagement_score_adjusted = np.clip(engagement_score + adjustment_factor + adjustment_factor_gaze, -1, 1)
    else:
        engagement_score_adjusted = engagement_score

    print("Old engagement score: ", engagement_score)  
    print("New engagement score: ", engagement_score_adjusted)  
    adjusted_engagement_scores.append((iterations, engagement_score_adjusted))

    image_path, iterations = engagement.generate_art(rec, engagement_score_adjusted, iterations, total_iterations, session_path)
    
    if iterations > total_iterations:
        return make_response('', 204)
        
    
    with open(image_path, 'rb') as image_file:
        img_data = image_file.read()

    if isinstance(engagement_score, (list, np.ndarray)):
        engagement_score = round(float(engagement_score[0]), 3)
    
    if iterations == total_iterations:
        scores_file_path = os.path.join(session_path, 'engagement_scores.csv')
        with open(scores_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Iteration', 'Engagement Score'])
            writer.writerows(engagement_scores)
        adj_file_path = os.path.join(session_path, 'adj_engagement_scores.csv')
        with open(adj_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Iteration', 'Engagement Score'])
            writer.writerows(adjusted_engagement_scores)
        session_path = None 
        data_to_send = {
            "img_data": img_data,
            "engagement_score": engagement_score,
            "end": True
        }
    else: 
        data_to_send = {
            "img_data": img_data,
            "engagement_score": engagement_score,
            "end": False
        }
    
    pickled_data = pickle.dumps(data_to_send)

    try:
        return send_file(
            io.BytesIO(pickled_data),
            as_attachment=True,
            download_name="data.pkl",
            mimetype="application/octet-stream"
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        

@app.route('/receive_score', methods=['POST'])
def submit_score():
    global session_path, iterations, engagement_scores, average_gaze, gaze_stddev
    iterations += 1
    gaze_scores.append(0)
    print("Flask iteration: ", iterations)
    
    if not request.json or 'score' not in request.json:
        return jsonify({"error": "No score provided"}), 400

    score = request.json['score']
    engagement_scores.append((iterations, score))
    adjusted_engagement_scores.append((iterations, score))
    
    # It should terminate here on the last one.
    image_path, iterations = engagement.generate_art(rec, score, iterations, total_iterations, session_path)
    
    if iterations > total_iterations:
        return make_response('', 204)
    
    with open(image_path, 'rb') as image_file:
        # Convert the image to bytes
        img_data = image_file.read()

    if iterations == total_iterations:
        scores_file_path = os.path.join(session_path, 'engagement_scores.csv')
        with open(scores_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Iteration', 'Engagement Score'])
            writer.writerows(engagement_scores)
        adj_file_path = os.path.join(session_path, 'adj_engagement_scores.csv')
        with open(adj_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Iteration', 'Engagement Score'])
            writer.writerows(adjusted_engagement_scores)
        session_path = None 
        data_to_send = {
            "img_data": img_data,
            "engagement_score": score,
            "end": True
        }
    else: 
        data_to_send = {
            "img_data": img_data,
            "engagement_score": score,
            "end": False
        }

    # Pickle the image bytes
    pickled_img_data = pickle.dumps(img_data)

    try:
        # Sending the pickled image data
        return send_file(
            io.BytesIO(pickled_img_data),
            as_attachment=True,
            download_name="pickled_image.pkl",
            mimetype="application/octet-stream"
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload_calibrate', methods=['POST'])
def upload_calibrate():
    global session_path, gaze_wrapper
    if 'calibrate_file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['calibrate_file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Date time and openface path
    openface_path = "docker exec openface_docker /home/openface-build/build/bin/"
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save the video within session folder for potential debug
    video_filename = f'calibration_video_{datetime_str}.mp4'
    video_filepath = os.path.join(session_path, video_filename)
    file.save(video_filepath)
    
    # A bunch of openface processing to get csv
    file.seek(0)
    videos_dir = f"/zooper2/{os.getenv('USER')}/openface_dump/received_videos"
    output_filename = f'video_{datetime_str}.mp4'
    output_filepath = os.path.join(videos_dir, output_filename)
    file.save(output_filepath)
    
    video_path = os.path.join("received_videos", output_filename)
    output_dir = "processed"
    
    command = f'{openface_path}FeatureExtraction -f {video_path} -out_dir {output_dir}'
    os.system(command)
    
    # Get the csv file   
    open_csv = os.path.join('/zooper2/colin.hwang/openface_dump/processed', f"{os.path.splitext(output_filename)[0]}.csv")
    
    # calibrate_folder = 'calibrate'
    # os.makedirs(calibrate_folder, exist_ok=True)
    # file_path = os.path.join(calibrate_folder, f'calibrate_{datetime_str}.csv')
    # Copy file over to desired directory
    # shutil.copy(open_csv, file_path)
    # file.save(file_path)
    gaze_wrapper.start(open_csv)
    #breakpoint()
    return jsonify({'message': 'File uploaded successfully', 'path': open_csv}), 200


if __name__ == '__main__':
    rec = ArtRecSystem(metric='cosine', art_generate=True)
    total_iterations = 25
    gaze_wrapper = GazeWrapper()
    engagement = Engagement()
    classifier = EngagementClassifierInference()
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S_EST")
    session_path = os.path.join('results', f'session_{date_str}')
    os.makedirs(session_path, exist_ok=True)
    os.makedirs(os.path.join(session_path, 'videos'), exist_ok=True)
    os.makedirs(os.path.join(session_path, 'art'), exist_ok=True)
    iterations = 0
    average_gaze = None
    gaze_stddev = None
    engagement_scores = []
    adjusted_engagement_scores = []
    gaze_scores = []
    
    app.run(host='0.0.0.0', port=8010)
