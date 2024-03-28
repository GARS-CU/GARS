import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class OpenFace:
    def __init__(self):
        self.openface_exe_path = r"C:\Users\Hcoli\OneDrive\Desktop\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"

    def extract_features(self, image_path, output_dir):
        command = f'"{self.openface_exe_path}" -f "{image_path}" -out_dir "{output_dir}"'
        os.system(command)

        # csv file name is image filename
        base_filename = os.path.basename(image_path)
        csv_filename = os.path.splitext(base_filename)[0] + ".csv"
        csv_path = os.path.join(output_dir, csv_filename)

        # load csv file. We are interested in the last 709 features
        data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
        features = data[1, 5:]

        return features

    def pca(self, features):
        features = np.reshape(features, (1, -1))

        scaler = StandardScaler()
        features_standardized = scaler.fit_transform(features)

        # reduce dim to 300
        pca = PCA(n_components=300)
        features_reduced = pca.fit_transform(features_standardized)

        # reshape for daisee regressor input
        features_pca= np.reshape(features_reduced, (300, 300, 1))

        return features_pca
