from flask import Flask, request, jsonify, send_file
import os
from datetime import datetime
from engagement_preprocess import Engagement
import requests
import sys
import io
import pickle
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Classes'))
from Daisee_Regressor_Final import EngagementClassifierInference

engagement = Engagement()
classifier = EngagementClassifierInference()

app = Flask(__name__)

@app.route('/process_data', methods=['POST'])
def process_data():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    videos_dir = '/zooper2/colin.hwang/openface_dump/received_videos'
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f'video_{datetime_str}.mp4'
    output_filepath = os.path.join(videos_dir, output_filename)
    file.save(output_filepath)
    
    openface_features = engagement.extract_open_features(output_filename)
    emotions_features = engagement.extract_emotion_features(output_filepath)
    
    open_features_pca = classifier.get_open_inference(openface_features)
    engagement_score = classifier.predict_engagement(emotions_features, open_features_pca)

    engagement_score = 2 * engagement_score - 1

    image_path = engagement.generate_art(engagement_score)

    with open(image_path, 'rb') as image_file:
        img_data = image_file.read()

    if isinstance(engagement_score, (list, np.ndarray)):
        engagement_score = round(float(engagement_score[0]), 3)

    data_to_send = {
        "img_data": img_data,
        "engagement_score": engagement_score
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
    if not request.json or 'score' not in request.json:
        return jsonify({"error": "No score provided"}), 400

    score = request.json['score']
    engagement = Engagement()
    image_path = engagement.generate_art(score)
    
    with open(image_path, 'rb') as image_file:
        # Convert the image to bytes
        img_data = image_file.read()

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



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8003)
