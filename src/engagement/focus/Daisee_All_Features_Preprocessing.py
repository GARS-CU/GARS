import os
import numpy as np
import pandas as pd
import tqdm
import sys
sys.path.append(os.environ['GARS_PROJ'])
from util import *

def export(dataset):

    path = os.path.join(var.GARS_PROJ, "datasets", "DAiSEE")

    df = pd.read_csv(os.path.join(path, "Labels", dataset + "Labels.csv"))

    labels = np.asarray(df["Boredom"])

    index = 0

    data = np.zeros((len(labels), 300, 709))

    dropped_indices = []
    
    for index, filename in enumerate(tqdm.tqdm(df["ClipID"].to_numpy())):
        open_face_file = os.path.join("..", "..", "datasets", "openface_csv_dump", filename[:-4] + ".csv")
        
        if os.path.exists(open_face_file):
            open_features = np.loadtxt(open_face_file, skiprows = 1,delimiter = ",")
            open_features = open_features[:,5:]
            #print(open_features.shape)
            #open_features = open_features[:5]
            if open_features.shape == (300, 709):
                data[index] = open_features
                continue
        
        print("Error: Removing " + open_face_file + " from list")
        dropped_indices.append(index)
    
    labels = np.delete(labels, dropped_indices, axis = 0)
    data = np.delete(data, dropped_indices, axis = 0)

    np.save(os.path.join(path, "DataSet", dataset, "openface_all_frames_features.npy"), data)
    np.save(os.path.join(path, "DataSet", dataset, "Engagement_Labels.npy"), labels)

export("Train")
export("Validation")

