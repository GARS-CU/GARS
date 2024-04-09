from flask import Flask, request, jsonify, send_file
import os
from datetime import datetime
from engagement_preprocess import Engagement
import requests
import sys
import io
import pickle
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Classes'))
from Daisee_Regressor_Final import EngagementClassifierInference

app = Flask(__name__)

@app.route('/process_data', methods=['POST'])
def process_data():
    engagement = Engagement()
    
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Ensure the directory for received videos exists
    videos_dir = '/app/openface_dump/received_videos'
    os.makedirs(videos_dir, exist_ok=True)

    # Use current datetime for a unique file name
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f'video_{datetime_str}.mp4'
    output_filepath = os.path.join(videos_dir, output_filename)

    # Save the video
    file.save(output_filepath)

    openface_service_url = "http://199.98.27.237:8001/process_openface"
    response = requests.post(openface_service_url, json={"filename": output_filename})
    if response.status_code != 200:
        return jsonify({"error": "Failed to process with OpenFace"}), 500

    # Send video functions to get openface features and emotions
    openface_features = engagement.extract_open_features(output_filename)
    emotions_features = engagement.extract_emotion_features(output_filepath)
    # print(emotions_features)
    
    classifier = EngagementClassifierInference()

    open_features_pca = classifier.get_open_inference(openface_features)
    print("Open Features PCA Shape:", openface_features.shape)

    engagement_score = classifier.predict_engagement(emotions_features, open_features_pca)
    print(engagement_score)

    return jsonify({"message": "Video processed and saved successfully", "engagement_score": float(engagement_score)}), 200

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
    app.run(host='0.0.0.0', port=8000)
