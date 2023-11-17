import tensorflow as tf
import pandas as pd
import cv2
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import layers


IMG_SIZE = -1

path_prefix = os.path.abspath("DAiSEE")

train = pd.read_csv(os.path.join(path_prefix, "Labels\TrainLabels.csv"))
test = pd.read_csv(os.path.join(path_prefix, "Labels\TestLabels.csv"))
val = pd.read_csv(os.path.join(path_prefix, "Labels\ValidationLabels.csv"))

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
            subset = "DataSet\Train"
            
        elif self.dataset == "val":
            data = val
            subset = "DataSet\Validation"
        else:
            data = test
            subset = "DataSet\Test"

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

        
inputs = keras.Input((10, 480, 640, 3))

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

outputs = layers.Dense(units = 4, activation = "softmax")(x)

model = keras.Model(inputs, outputs)


model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer = keras.optimizers.Adam(learning_rate = 1e-3),
              metrics = ["acc"])

epochs = 50

model.fit(train_ds,
          validation_data = val_ds,
          epochs = epochs,
          batch_size = 32
          )