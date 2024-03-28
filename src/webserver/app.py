from flask import Flask, request, jsonify
import os
from datetime import datetime
import pickle
from PIL import Image
import io
import numpy as np

app = Flask(__name__)

@app.route('/process_data', methods=['POST'])
def process_data():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Read the pickled data from the file
    pickled_data = file.read()
    
    # Unpickle img and convert to img object
    image_data = pickle.loads(pickled_data)
    image = Image.open(io.BytesIO(image_data))

    # use time date for unique file name
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f'frame_{datetime_str}.jpg'
    output_filepath = os.path.join('frames', output_filename)

    # save img
    image.save(output_filepath)

    return jsonify({"message": "Image processed and saved successfully"}), 200

@app.route('/process_pca_features', methods=['POST'])
def process_pca_features():
    # get openface feature payload
    data = request.json
    features = data.get('features')

    if not features:
        return jsonify({"error": "Missing features"}), 400

    # convert to np array
    features_array = np.array(features).reshape((300, 300, 1))

    # debug print and success msg
    print("Received PCA features with shape:", features_array.shape)
    return jsonify({"message": "PCA features received and reshaped successfully"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
