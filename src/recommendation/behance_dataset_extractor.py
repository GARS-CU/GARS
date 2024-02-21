import struct
import numpy as np
from tqdm import tqdm
import pickle
import os
import sys
sys.path.insert(0, os.path.abspath(".."))
sys.path.append(os.environ['GARS_PROJ'])
from util import *

#we store the features for each item in a numpy array
item_features = np.zeros((178787, 4096))

#we map the item id to the correspoding index in the numpy array
item_index = {}

#path to the behance dataset files
rec_path = os.path.join(var.GARS_PROJ, "datasets", "Behance")

#we load in the item features into our numpy array
with open(os.path.join(rec_path, "Behance_Image_Features.b"), "rb") as f:
    
    for index in tqdm(range(len(item_features))):
        
        itemId = f.read(8)

        if itemId == '': break
        
        feature = struct.unpack('f'*4096, f.read(4*4096))

        item_features[index] = feature
        item_index[itemId] = index

np.save(os.path.join(rec_path, "behance_item_features.npy"), item_features)

#we map the user id to the list of indices in the item numpy array corresponding 
#to the items that the user appreciated
user_index = {}

with open(os.path.join(rec_path, "Behance_appreciate_1M")) as f:
    for line in f:
        ids = line.split(" ")
        if ids[0] not in user_index:
            user_index[ids[0]] = []
        if ids[1] != "01398047":
            user_index[ids[0]].append(item_index[ids[1].encode("utf-8")])

with open(os.path.join(rec_path, "Behance_user_item_appreciates.pkl"), "wb") as f:
    pickle.dump(user_index, f)
