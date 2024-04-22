from flask import Flask, render_template, request, jsonify, send_from_directory
import requests
import os
from datetime import datetime
from PIL import Image
import io
import pickle
import glob

app = Flask(__name__)

video_files_directory = os.path.join(os.getcwd(), 'video_files')
os.makedirs(video_files_directory, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    video = request.files['video']
    video_data = video.read()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    url = 'http://199.98.27.237:8003/process_data'
    files = {'file': ('video.mp4', video_data, 'video/mp4')}
    response = requests.post(url, files=files)

    if response.status_code == 200:
        # Unpickle the received data to get both the image bytes and engagement score
        received_data = pickle.loads(response.content)
        img_data = received_data['img_data']
        engagement_score = received_data['engagement_score']

        generated_art_dir = 'static/assets/Generated_Art'
        os.makedirs(generated_art_dir, exist_ok=True)

        # Save the image data to a file
        image_path = os.path.join(generated_art_dir, f"generated_art_{timestamp}.png")
        with open(image_path, 'wb') as f:
            f.write(img_data)

        return jsonify({
            "status": "success",
            "message": "Art and engagement score received successfully",
            "imagePath": image_path,
            "engagementScore": engagement_score
        })
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

    url = 'http://199.98.27.237:8003/receive_score'
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

if __name__ == '__main__':
    app.run(debug=True)