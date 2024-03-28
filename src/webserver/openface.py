import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import pickle as pk

OPENFACE_PATH = "sudo docker exec openface_docker /home/openface-build/build/bin/"

class OpenFace:

    def extract_features_from_image(self, image_path):
        output_dir = "processed"
        os.system(f'{OPENFACE_PATH}FaceLandmarkImg -f "{image_path}" -out_dir {output_dir}')

        feat_file = os.path.join(output_dir, os.path.basename(image_path)[:-4] + ".csv")

        features = np.genfromtxt(feat_file, delimiter=",", skip_header=1)

        # Additional processing here if necessary
        return features

    def process_features(self, features):
        """
        Processes the extracted features using MinMaxScaler and PCA, similar to get_open.
        """
        scaler = MinMaxScaler()
        pca = PCA(n_components=300)

        # Assuming features is already in the correct shape; adjust if necessary
        features = scaler.fit_transform(features)
        features_transformed = pca.fit_transform(features)

        # Save PCA model
        pca_model_path = "pca_model.pkl"  # Adjust path as needed
        pk.dump(pca, open(pca_model_path, "wb"))

        # Reshape for the model
        features_transformed = np.reshape(features_transformed, (-1, 300, 300, 1))

        return features_transformed

