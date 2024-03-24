import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.append(os.path.abspath("../.."))
sys.path.append(os.environ['GARS_PROJ'])
from util import *


df = pd.read_csv(os.path.join(var.GARS_PROJ, "datasets", "FERP", "FER_plus.csv"))

dataset = np.zeros((35887, 48, 48, 1))
labels = []
count = 0

for index, row in df.iterrows():

    if row["NF"] == 0:
        dataset[count] = np.reshape(np.array(row["pixels"].split(" "), dtype=  float), (48, 48, 1))
        labels.append([df.iloc[index, categ]/10 for categ in range(3, 13)])
        count += 1

dataset = dataset[:count]
labels = np.array(labels)

x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size = .2)
x_val, x_test, y_val, y_test = train_test_split(dataset, labels, test_size = .2)

print(y_val.shape)
                  
np.save(os.path.join(var.GARS_PROJ, "datasets", "FERP", "FERP_xtrain.npy"), x_train)
np.save(os.path.join(var.GARS_PROJ, "datasets", "FERP", "FERP_ytrain.npy"), y_train)
np.save(os.path.join(var.GARS_PROJ, "datasets", "FERP", "FERP_xtest.npy"), x_test)
np.save(os.path.join(var.GARS_PROJ, "datasets", "FERP", "FERP_ytest.npy"), y_test)
np.save(os.path.join(var.GARS_PROJ, "datasets", "FERP", "FERP_xval.npy"), x_val)
np.save(os.path.join(var.GARS_PROJ, "datasets", "FERP", "FERP_yval.npy"), y_val)
