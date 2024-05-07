from flask import Flask, render_template, request, jsonify, send_from_directory
import requests
import os
from datetime import datetime
from PIL import Image
import io
import pickle
import glob
import numpy as np

app = Flask(__name__)

video_durations = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    video = request.files['video']
    video_data = video.read()
    duration = float(request.form['duration'])
    video_durations.append(duration)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    url = 'http://199.98.27.237:8010/process_data'

    files = {'file': ('video.mp4', video_data, 'video/mp4')}
    data = {}

    if len(video_durations) >= 5:
        first_durations = video_durations[:5]
        average_duration = np.mean(first_durations)
        std_deviation = np.std(first_durations)
        
        data['average_duration'] = average_duration
        data['std_deviation'] = std_deviation 
        data['duration'] = duration

    response = requests.post(url, files=files, data=data)

    if response.status_code == 200:
        # Unpickle the received data to get both the image bytes and engagement score
        received_data = pickle.loads(response.content)
        img_data = received_data['img_data']
        engagement_score = received_data['engagement_score']
        session_end = received_data.get('end', False) 

        generated_art_dir = 'static/assets/Generated_Art'
        os.makedirs(generated_art_dir, exist_ok=True)

        # Save the image data to a file
        image_path = os.path.join(generated_art_dir, f"generated_art_{timestamp}.png")
        with open(image_path, 'wb') as f:
            f.write(img_data)

        response_data = {
            "status": "success",
            "message": "Art and engagement score received successfully",
            "imagePath": image_path,
            "engagementScore": engagement_score,
            "sessionEnd": session_end
        }

        if 'average_duration' in data:
            response_data['averageDuration'] = data['average_duration']
            response_data['stdDeviation'] = data['std_deviation']    

        return jsonify(response_data)
    
    else:
        return jsonify({"status": "failure", "message": "Failed to receive art"})

@app.route('/submit_score', methods=['POST'])
def submit_score():
    score = request.form['score']
    print("Score received:", score)
    try:
        score_val = float(score)
    except ValueError:
        return jsonify({"status": "failure", "message": "Invalid score format"})

    if not -1 <= score_val <= 1:
        return jsonify({"status": "failure", "message": "Score must be between -1 and 1"})

    url = 'http://199.98.27.237:8010/receive_score'
    payload = {'score': score_val}
    headers = {'Content-Type': 'application/json'}

    response = requests.post(url, json=payload, headers=headers, stream=True)

    if response.status_code == 200:
        generated_art_dir = 'static/assets/Generated_Art'
        os.makedirs(generated_art_dir, exist_ok=True)

        # Unpickle the received data to get the raw image bytes
        img_data = pickle.loads(response.content)

        # Save the image data to a file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(generated_art_dir, f"generated_art_{timestamp}.png")
        with open(image_path, 'wb') as f:
            f.write(img_data)

        return jsonify({"status": "success", "message": "Art received successfully", "imagePath": image_path})
    else:
        return jsonify({"status": "failure", "message": "Failed to receive art"})

@app.route('/calibrate', methods=['GET', 'POST'])
def calibrate():
    if request.method == 'POST':
        video = request.files['video']
        video_data = video.read()

        files = {'calibrate_file': ('calibrate_file.mp4', video_data, 'video/mp4')}
        response = requests.post('http://199.98.27.237:8010/upload_calibrate', files=files)

        if response.status_code == 200:
            return jsonify({"status": "success", "message": "Calibration video uploaded successfully"})
        else:
            return jsonify({"status": "failure", "message": "Failed to upload calibration video"})

    return render_template('calibrate.html')



if __name__ == '__main__':
    app.run(debug=True)