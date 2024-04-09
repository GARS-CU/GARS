from flask import Flask, render_template, request, jsonify
import requests
import os
from datetime import datetime

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

    # save the video for debugging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"video_{timestamp}.mp4"
    video_path = os.path.join(video_files_directory, video_filename)
    with open(video_path, 'wb') as vid_file:
        vid_file.write(video_data)

    url = 'http://199.98.27.237:8000/process_data'
    files = {'file': ('video.mp4', video_data, 'video/mp4')}
    response = requests.post(url, files=files)

    if response.status_code == 200:
        return jsonify({"status": "success", "message": "Video processed and sent successfully"})
    else:
        return jsonify({"status": "failure", "message": "Failed to send video"})

if __name__ == '__main__':
    app.run(debug=True)
