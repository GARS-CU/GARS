import tensorflow as tf
import pandas as pd
import cv2
import numpy as np
import os
import sys
sys.path.append(os.environ['GARS_PROJ'])
from util import *
from tensorflow import keras
from tensorflow.keras import layers

"""
IMG_SIZE = -1

path_prefix = os.path.abspath(os.path.join("..", "..", "datasets", "DAiSEE"))

train = pd.read_csv(os.path.join(path_prefix, "Labels", "TrainLabels.csv"))
test = pd.read_csv(os.path.join(path_prefix, "Labels", "TestLabels.csv"))
val = pd.read_csv(os.path.join(path_prefix, "Labels", "ValidationLabels.csv"))

train_paths = train["ClipID"]
val_paths = val["ClipID"]
test_paths = test["ClipID"]


def load_video(path, frame_rate = 30):


    cap = cv2.VideoCapture(path)

    frames = []

    count = 0
    try:
        while count < frame_rate*10:
            ret, frame = cap.read()
            if not ret:
                break
            #frame = [:, :, [2,1,0] ]
            frames.append(frame)

            count += frame_rate

            cap.set(cv2.CAP_PROP_POS_FRAMES, count)

    
    finally:
        cap.release()
    
    return np.array(frames).reshape(1, 10, 480, 640, 3)



class FrameGenerator:
    def __init__(self, dataset, training = False):

        self.dataset = dataset
        self.training = training
    
    def __call__(self):
        
        if self.dataset == "train":
            data = train
            subset = os.path.join("DataSet", "Train")
            
        elif self.dataset == "val":
            data = val
            subset = os.path.join("DataSet", "Validation")
        else:
            data = test
            subset = os.path.join("DataSet", "Test")

        if self.training:
            data = data.sample(frac = 1).reset_index(drop = True)

        fnames = data["ClipID"]

        labels = np.asarray(data["Engagement"])
        labels = labels.reshape( (1, len(labels)))
        for idx in range(len(fnames)):

            path = os.path.join(path_prefix, subset)
            path = os.path.join(path, fnames.iloc[idx][:6])
            path = os.path.join(path, fnames.iloc[idx][:-4])
            path = os.path.join(path, fnames.iloc[idx])

            yield load_video(path), np.asarray(labels[:, idx]).reshape(1, 1)
        

output_signature = (tf.TensorSpec(shape = (None, 10, 480, 640, 3), dtype = tf.float32),
                    tf.TensorSpec(shape = (None, 1), dtype = tf.int16))


train_ds = tf.data.Dataset.from_generator(FrameGenerator("train", training = True),
                                          output_signature = output_signature)

val_ds = tf.data.Dataset.from_generator(FrameGenerator("val"),
                                        output_signature = output_signature)

test_ds = tf.data.Dataset.from_generator(FrameGenerator("test"),
                                        output_signature = output_signature)

"""

path_prefix = os.path.join(var.GARS_PROJ, "datasets", "DAiSEE", "DataSet")

x_train = np.load(os.path.join(path_prefix, "Train", "emotion_large.npy"))/255.0
y_train = np.load(os.path.join(path_prefix, "Train", "labels.npy"))

indices = np.array(list(range(len(x_train))))
indices = np.random.choice(indices, size = int(0.2*len(x_train)), replace = False)
x_val = x_train[indices]
y_val = y_train[indices]
x_train = np.delete(x_train, indices, axis = 0)
y_train = np.delete(y_train, indices, axis = 0)
#x_val = np.load(os.path.join(path_prefix, "Validation", "emotion_large.npy"))
#y_val = np.load(os.path.join(path_prefix, "Validation", "labels.npy"))


y_train[y_train == 3] = 2
y_val[y_val == 3] = 2

inputs = keras.Input((10, 100, 100, 1))

x = layers.Conv3D(filters = 64, kernel_size = 3, activation = "relu", padding = "SAME")(inputs)
x = layers.AveragePooling3D(pool_size = 2)(x)
x = layers.BatchNormalization()(x)

x = layers.Conv3D(filters = 64, kernel_size = 3, activation = "relu", padding = "SAME")(x)
x = layers.AveragePooling3D(pool_size = 2)(x)
x = layers.BatchNormalization()(x)

x = layers.Conv3D(filters = 128, kernel_size = 3, activation = "relu", padding = "SAME")(x)
x = layers.AveragePooling3D(pool_size = 2)(x)
x = layers.BatchNormalization()(x)

x = layers.GlobalAveragePooling3D()(x)
x = layers.Dense(units=512, activation="relu")(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(units = 3, activation = "softmax")(x)

model = keras.Model(inputs, outputs)

model.summary()
model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer = keras.optimizers.Adam(learning_rate = 1e-3),
              metrics = ["acc"])

epochs = 300

model.fit(x_train, y_train,
          validation_data = (x_val, y_val),
          epochs = epochs,
          batch_size = 32
          )
