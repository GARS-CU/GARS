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
import sys
sys.path.append(os.environ['GARS_PROJ'])
from util import *

class Custom_Attention(tf.keras.layers.Layer):

        def call(self, x):
            size = tf.shape(x)
            x = tf.reshape(x, [size[0], size[1], 1])
            y = tf.einsum("...ij,...jk->...ik", x, tf.transpose(x, perm = [0, 2, 1]))/16.0
            y = tf.matmul(tf.nn.softmax(y), x)
            return tf.reshape(y, [size[0], size[1]])
        

#we read in the labels of the split video clips

df = pd.read_csv(os.path.join(var.GARS_PROJ, "datasets", "EmotiW", "Engagement_Labels_Split.csv"))

path_prefix = os.path.abspath(os.path.join(var.GARS_PROJ, "datasets", "EmotiW"))
    
#in this function we sample 10 frames from the 10 second clip
#with one frame being sampled per second
def load_video(path, frame_rate = 30):


    cap = cv2.VideoCapture(path)

   
    frames = []

    count = 0
    try:
        while count < frame_rate*10:
            ret, frame = cap.read()
            
            #we gray scale the image
            frame = cv2.cvtColor(frame, cv2.IMREAD_GRAYSCALE)
            if not ret:
                break
            #and the convert it to 1 channel
            frame  = np.mean(frame, axis = -1)

            #the faces in the emotiw dataset are centered so
            #we can reduce the image size by cropping out
            #for the face
            center = frame.shape
            x = int(center[1]/2 - 100)
            y = int(center[0]/2 - 100)

            frame = frame[y:y+200, x:x+200]

            #we then downsample into a 48x48 image. This results
            #in a serious decrease in quality that may affect our model's
            #performance
            frame = cv2.resize(frame, (48, 48))
            cv2.imshow("image", frame)
            frames.append(frame)

            count += frame_rate

            cap.set(cv2.CAP_PROP_POS_FRAMES, count)

    
    finally:
        cap.release()
    
    #we then stack the 10 frames collected into a single array
    return np.array(frames).reshape(1, 10, 48, 48, 1)


#Since we're using a dataset of videos, the dataset is very large
#and so we can't really store it in a single numpy array. Instead
#we can generate the frames as needed when training our model
class FrameGenerator:
    
    #constructor for the FrameGenerator, takes in dataset, a string
    #that indicates what set to draw the data from and training,
    #a boolean indiciating whether or not to shuffle the dataset
    def __init__(self, dataset, training = False):
        self.dataset = dataset
        self.training = training
    
    def __call__(self):
        
        #we get the filenames of the files in the corresponding set
        data = df.loc[df["Dataset"] == self.dataset]
    
        #if we're training then we shuffle the list of files
        if self.training:
            data = data.sample(frac = 1).reset_index(drop = True)

        #and then iterate through the list
        for _, row in data.iterrows():
            #and yield a stack of frames to be used as input to the model
            #as well as the label for the stack of frames
            yield load_video(row["Path"]), np.asarray(row["Score"]).reshape(1, 1)

output_signature = (tf.TensorSpec(shape = (None, 10, 48, 48, 1), dtype = tf.float32),
                    tf.TensorSpec(shape = (None, 1), dtype = tf.float32))


#create our training and validation set
train_ds = tf.data.Dataset.from_generator(FrameGenerator("Train", training = True),
                                          output_signature = output_signature)

val_ds = tf.data.Dataset.from_generator(FrameGenerator("Validation"),
                                        output_signature = output_signature)


 
#we load in the weights of our emotion recognition classifier
emoti_model = Patt_Lite()

tf.keras.utils.get_custom_objects()["Custom_Attention"] = Custom_Attention

emoti_model.model.load_weights(os.path.join(var.GARS_PROJ, "Models", "Emption_Rec", "PAtt_Lite_weights.h5"))

#unfortunately we weren't really able to find any public engagement datasets
#that for frame level classification. Instead all of the datasets we found
#were about classifying a person's engagement based on a video. This is problematic
#because our emotion recognition and haar classifier work on the level of frames.
#We couldn't use single frames from the video because they wouldn't be representative
#of the entire video and so the same label couldn't be applied to them. So instead
#we decided to use the average engagement as the output of our model. We feed in 
#several sampled frames into our emotion rec clsasifier and then use the averaged
#output of our emotion rec classifier as the output of our entire model (after passing
#it through a dense layer)
class AggregationLayer(Layer):

    def __init__(self, emo_model, num_frames, **kwargs):
        super(AggregationLayer, self).__init__(**kwargs)
        self.emoti_model = emo_model
        self.num_frames = num_frames
    
    def call(self, inputs):
        frame_outputs = [self.emoti_model(frame) for frame in tf.unstack(inputs, axis=1)]
        averaged_output = tf.reduce_mean(tf.stack(frame_outputs), axis=0)
        return averaged_output

inputs = keras.Input((10, 48, 48, 1))

#we calculate the average output
x = AggregationLayer(emoti_model.model, 10)(inputs)

#and then feed it into a dense layer to get our engagement score
outputs = Dense(1, activation = "relu")(x)

model = Model(inputs, outputs)

model.summary()

model.compile(optimizer='adam', loss='mse')

model.fit(train_ds,
          validation_data = val_ds,
          epochs = 10,
          batch_size = 32
          ) 
