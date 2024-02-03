from sklearn.decomposition import PCA
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import os
import sys
sys.path.append(os.environ['GARS_PROJ'])
from util import *

def process_dataset():
    path = os.path.join(var.GARS_PROJ, "datasets", "DAiSEE", "DataSet")
    
    scaler = MinMaxScaler()
    
    #load in the labels and feature sets
    y_train =  np.load(os.path.join(path, "Train", "Engagement_Labels.npy"))
    
    
    x_train = np.load(os.path.join(path, "Train", "openface_all_frames_features.npy"))
    
    
    #normalize the feature set
    x_train = scaler.fit_transform(np.reshape(x_train, (-1, 709)))
    
    #apply pca to get a 300x300 array of features
    pca = PCA(n_components = 300)
    x_train = pca.fit_transform(x_train)
    
    x_train = np.reshape(x_train, (len(y_train), 300, 300, 1))
    
    #do the same for the validation set
    y_val = np.load(os.path.join(path, "Validation", "Engagement_Labels.npy"))

    x_val = np.load(os.path.join(path, "Validation", "openface_all_frames_features.npy"))
    x_val = scaler.transform(np.reshape(x_val, (-1, 709)))

    
    x_val = np.reshape(pca.transform(x_val), (len(y_val), 300, 300, 1))
    
    #then we rebalance the dataset by only sampling as many samples from each class that 
    #are in the smallest class
    """
    balance_size = min([np.sum(y_train == categ) for categ in range(4)])
    
    
    indices =[]
    for categ in range(4):
        ind = np.where(y_train == categ)[0]
        indices += np.random.choice(ind, size = balance_size, replace = False).tolist()
    
    
    x_train  = x_train[indices]
    y_train = y_train[indices]
    """
    return x_train, y_train, x_val, y_val

#get the train and validation sets
x_train, y_train, x_val, y_val = process_dataset()

#create our model
inputs = keras.layers.Input(shape = (300, 300, 1))

x = keras.layers.Conv2D(32, 5, activation = "relu")(inputs)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(64, 5, activation = "relu")(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(128, 5, activation = "relu")(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(256, 5, activation = "relu")(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Dropout(0.25)(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(128, activation = "relu")(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(4, activation = "softmax")(x)

model = keras.models.Model(inputs,  x)

model.summary()
model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer = keras.optimizers.Adam(learning_rate = 1e-3),
              metrics = ["acc"])

model.fit(x = x_train,
          y = y_train,
          validation_data = (x_val, y_val),
          epochs = 1600,
          batch_size = 8
          )

