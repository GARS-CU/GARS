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

class Custom_Attention(tf.keras.layers.Layer):

        def call(self, x):
            size = tf.shape(x)
            x = tf.reshape(x, [size[0], size[1], 1])
            y = tf.einsum("...ij,...jk->...ik", x, tf.transpose(x, perm = [0, 2, 1]))/16.0
            y = tf.matmul(tf.nn.softmax(y), x)
            return tf.reshape(y, [size[0], size[1]])
        
""" 
df = pd.read_csv("..\..\datasets\EmotiW\Engagement_Labels_Split.csv")#, names = ["Filename", "Scores", "Dataset"], header = None)


path_prefix = os.path.abspath("..\..\datasets\EmotiW")
    

def load_video(path, frame_rate = 30):


    cap = cv2.VideoCapture(path)

   
    frames = []

    count = 0
    try:
        while count < frame_rate*10:
            ret, frame = cap.read()

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            if not ret:
                break

            center = frame.shape
            x = int(center[1]/2 - 100)
            y = int(center[0]/2 - 100)

            frame = frame[y:y+200, x:x+200]
            #frame = [:, :, [2,1,0] ]

            frame = cv2.resize(frame, (48, 48))
            cv2.imshow("image", frame)
            frames.append(frame)

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
        
        
        data = df.loc[df["Dataset"] == self.dataset]
                      
        if self.training:
            data = data.sample(frac = 1).reset_index(drop = True)


        #labels = np.asarray(data["Engagement"])
        #labels = labels.reshape( (1, len(labels)))
        for row, _ in data.iterrows():

            yield load_video(row["Path"]), np.asarray(row["Scores"]).reshape(1, 1)

output_signature = (tf.TensorSpec(shape = (None, 10, 48, 48, 3), dtype = tf.float32),
                    tf.TensorSpec(shape = (None, 1), dtype = tf.float32))


train_ds = tf.data.Dataset.from_generator(FrameGenerator("Train", training = True),
                                          output_signature = output_signature)

val_ds = tf.data.Dataset.from_generator(FrameGenerator("Validation"),
                                        output_signature = output_signature)


 """

emoti_model = Patt_Lite()

tf.keras.utils.get_custom_objects()["Custom_Attention"] = Custom_Attention

emoti_model.model.load_weights("../../Models/Emotion_Rec/PAtt_Lite_weights.h5")
emoti_model.model.summary()
x_train = np.load("../../datasets/FERP/FERP_xtrain.npy")
y_train = np.load("../../datasets/FERP/FERP_ytrain.npy")
y_train = np.argmax(y_train, axis = 1)

y_pred = emoti_model.model.predict(x_train)

print(y_pred)
print(y_train)
""" class AggregationLayer(Layer):

    def __init__(self, emo_model, num_frames, **kwargs):
        super(AggregationLayer, self).__init__(**kwargs)
        self.emoti_model = emoti_model
        self.num_frames = num_frames
    
    def call(self, inputs):
        frame_outputs = [self.emo_model(frame) for frame in tf.unstack(inputs, axis=1)]
        averaged_output = tf.reduce_mean(tf.stack(frame_outputs), axis=0)
        return averaged_output

inputs = keras.Input((10, 48, 48, 1))

x = AggregationLayer(emoti_model, 10)(inputs)

outputs = Dense(1, activation = "relu")(x)

model = Model(inputs, outputs)

model.compile(optimizer='adam', loss='mse')

model.fit(train_ds,
          validation_data = val_ds,
          epochs = 10,
          batch_size = 32
          ) """