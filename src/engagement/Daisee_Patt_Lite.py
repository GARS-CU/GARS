import numpy as np
import cv2
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Dense
from keras import models
from PAtt_Lite import Patt_Lite
from deepface import DeepFace
import contextlib
import sys
sys.path.append(os.environ['GARS_PROJ'])
from util import *
face_cascade = cv2.CascadeClassifier(os.path.join(var.GARS_PROJ, "haarcascade_frontalface_alt.xml"))


class Custom_Attention(tf.keras.layers.Layer):

        def call(self, x):
            size = tf.shape(x)
            x = tf.reshape(x, [size[0], size[1], 1])
            y = tf.einsum("...ij,...jk->...ik", x, tf.transpose(x, perm = [0, 2, 1]))/16.0
            y = tf.matmul(tf.nn.softmax(y), x)
            return tf.reshape(y, [size[0], size[1]])
        

IMG_SIZE = -1

path_prefix = os.path.abspath(os.path.join(var.GARS_PROJ, "datasets", "DAiSEE"))

train = pd.read_csv(os.path.join(path_prefix, "Labels" , "TrainLabels.csv"))
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

            processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            face_locations = face_cascade.detectMultiScale(frame,scaleFactor = 1.05,minNeighbors=5)
            if len(face_locations) == 0:
                with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):

                    processed_frame = DeepFace.extract_faces(frame, detector_backend = "mtcnn", target_size = (100,100), grayscale = True)
                processed_frame = max(processed_frame, key = lambda x: x["facial_area"]["w"]*x["facial_area"]["h"])["face"]
            else:
                face_location = [max(face_locations, key = lambda x: x[2]*x[3])]

                for x,y,h,w in face_location:
                    processed_frame = processed_frame[y:y+h, x:x+w]
            

                #processed_frame = cv2.resize(processed_frame,(int(frame.shape[1]/2) , int(frame.shape[0]/2)))

            processed_frame = cv2.resize(processed_frame,(48 , 48))

            frames.append(processed_frame)

            count += frame_rate

            cap.set(cv2.CAP_PROP_POS_FRAMES, count)
    
    finally:
        cap.release()
    
    return np.array(frames).reshape(1, 10, 48, 48, 1)



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
        

output_signature = (tf.TensorSpec(shape = (None, 10, 48, 48, 1), dtype = tf.float32),
                    tf.TensorSpec(shape = (None, 1), dtype = tf.int16))


train_ds = tf.data.Dataset.from_generator(FrameGenerator("train", training = True),
                                          output_signature = output_signature)

val_ds = tf.data.Dataset.from_generator(FrameGenerator("val"),
                                        output_signature = output_signature)

test_ds = tf.data.Dataset.from_generator(FrameGenerator("test"),
                                        output_signature = output_signature)



emoti_model = Patt_Lite().model

tf.keras.utils.get_custom_objects()["Custom_Attention"] = Custom_Attention

emoti_model.load_weights(os.path.join(var.GARS_PROJ, "Models", "Emotion_Rec", "PAtt_Lite_weights.h5"))


class AggregationLayer(Layer):

    def __init__(self, emo_model, num_frames, **kwargs):
        super(AggregationLayer, self).__init__(**kwargs)
        self.emoti_model = emo_model
        self.num_frames = num_frames
    
    def __call__(self, inputs):
        frame_outputs = [self.emoti_model(frame) for frame in tf.unstack(inputs, axis=1)]
        averaged_output = tf.reduce_mean(tf.stack(frame_outputs), axis=0)
        print(averaged_output)
        return averaged_output
    
inputs = keras.Input((10, 48, 48, 1))

y = AggregationLayer(emoti_model, 10)(inputs)

outputs = Dense(4, activation = "softmax")(y)

model = Model(inputs, outputs)

model.summary()

model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer = keras.optimizers.Adam(learning_rate = 1e-3),
              metrics = ["acc"])

model.fit(train_ds,
          validation_data = val_ds,
          epochs = 10,
          batch_size = 32
          )

