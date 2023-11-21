import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("FER_plus.csv")

dataset = np.zeros((35887, 48, 48, 1))
labels = []
count = 0

for index, row in df.iterrows():

    if row["NF"] == 0:
        dataset[count] = np.reshape(np.array(row["pixels"].split(" "), dtype=  float), (48, 48, 1))
        labels.append([df.iloc[index, categ]/10 for categ in range(4, 13)])
        count += 1

dataset = dataset[:count]
labels = np.array(labels)

x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size = .2)
x_val, x_test, y_val, y_test = train_test_split(dataset, labels, test_size = .2)
                  
np.save( "FERP_xtrain.npy", x_train)
np.save("FERP_ytrain.npy", y_train)
np.save("FERP_xtest.npy", x_test)
np.save("FERP_ytest.npy", y_test)
np.save("FERP_xval.npy", x_val)
np.save("FERP_yval.npy", y_val)