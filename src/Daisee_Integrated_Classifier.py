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

#This program implements and trains the complete engagement detection model (emotion + focus classifier) on the DAiSEE dataset

#get the train, val, and test dataframes containing the paths to the video files in each dataset as well as the labels
path_prefix = os.path.abspath("..\..\datasets\DAiSEE")

train = pd.read_csv(os.path.join(path_prefix, "Labels\TrainLabels.csv"))
test = pd.read_csv(os.path.join(path_prefix, "Labels\TestLabels.csv"))
val = pd.read_csv(os.path.join(path_prefix, "Labels\ValidationLabels.csv"))

#The Boredom labels were a lot more balanced than the Engagement labels so I thought it might be good to use them instead
#We can switch over to engagement though and make the dataset more balanced if needed

#There weren't that many videos that were classified as extremely bored so I lumped the two highest boredom scores together
train.loc[train["Boredom"] == 3, "Boredom"] = 2



#we load in the haar cascade classifier used for face detection. This model has some trouble with detecting 
#faces when they're rotated so if it can't detect a face, I used a more intensive neural network from the deepface library to crop the face.
#The bigger model takes around 30 seconds to evaluate so I only used it if the haar cascade model couldn't find anything.
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

#This function uses OpenFace to extract high level facial features from the video. Takes in the 
#path to the video file. Returns a 10x35 numpy array with each row representing the averaged facial features
#for each second which is used as the input for our focus classifier

def gen_features(video_path):

    #we run OpenFace's FeatureExtraction executable which extracts the features for each frame in the video and exports them to a csv file.
    #I tried to see if there way to extract the features for one frame every second but I couldn't find a good way to do that.
    #In addition to the csv file, it also creates a video file which isn't really necessary either. It ends up taking around 6 seconds on my computer to 
    #extract the features for a 10 second video which is pretty intensive.

    #Right now, it only extracts the 35 action units but we could do things like head pose and eye gaze instead
    os.system(".\..\..\Models\OpenFace_2.2.0_win_x64\FeatureExtraction > $null -f " + video_path + " -aus")

    #we get the name of the csv file that the features are stored in
    feat_file = os.path.join("processed", os.path.basename(video_path)[:-3] + "csv")

    #and load it into a numpy array
    data = np.genfromtxt(feat_file, invalid_raise = False, delimiter = ", ", skip_header = 1)[:300]

    #the first 5 columns contain things like frame # and time stamp which aren't necessary
    data = data[:,5:]
    
    #we fill in any nan values with the averaged feature value over the 10 seconds (I haven't seen any nan values
    #so far in this dataset but the open face features in the emotiw dataset did have some so this is just in case)
    col_mean = np.nanmean(data, axis = 0)
    indices = np.where(np.isnan(data))

    data[indices] = np.take(col_mean, indices[1])

    #The frame rate for the video is 30 fps so for each second, we average the extracted features over the 30 frames
    data = data.reshape(10, 30, 35).mean(axis = 1)

    #Finally, we remove the folder containing the csv file
    os.system("rd /s /q processed")

    return data

#This function processes the given video file and returns the inputs to the emotion and focus classifier
#In the emotion classifier, a face detection model is used to crop the face and then downsample to a 48x48
#greyscaled image. We were worried that the image quality would suffer from this but after cropping, the size of the face
#is typically only around 100x100 so the quality doesn't change much. Takes in the path to the video file and the frame rate
#of the video

def load_video(path, frame_rate = 30):

    cap = cv2.VideoCapture(path)

    frames = []

    count = 0
    try:
        #as each video is 10 seconds long, we sample a frame for each second
        while count < frame_rate*10:
            ret, frame = cap.read()
            if not ret:
                break
            #we gray scale the image
            processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #then we use the haar cascade model to find where the face is located
            face_locations = face_cascade.detectMultiScale(frame,scaleFactor = 1.05,minNeighbors=5)

            #if no face was found
            if len(face_locations) == 0:
                    
                    #we use the MTCNN model in the deepface library to extract face
                    try:
                        #suppress prints
                        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                            processed_frame = DeepFace.extract_faces(frame, detector_backend = "mtcnn", target_size = (100,100), grayscale = True)
                        
                        #if multiple faces are found, we take the largest one
                        processed_frame = max(processed_frame, key = lambda x: x["facial_area"]["w"]*x["facial_area"]["h"])["face"]

                    except:
                        #and if that doesn't work then we give a blank image for that frame (this rarely happens)
                        processed_frame = np.zeros((100, 100))
            else:
                #if multiple faces are found, we take the largest one
                face_location = [max(face_locations, key = lambda x: x[2]*x[3])]

                for x,y,h,w in face_location:
                    #we then crop out the face
                    processed_frame = processed_frame[y:y+h, x:x+w]
            

            #we downsample the image to a 48x48 array
            processed_frame = cv2.resize(processed_frame,(48 , 48))

            #and append the processed frame to our list
            frames.append(processed_frame)

            count += frame_rate
        
            cap.set(cv2.CAP_PROP_POS_FRAMES, count)
    
    finally:
        cap.release()
    
    #we get the open face features
    open_face = gen_features(path)
    
    #and then return a tuple with the first entry being the input to our emotion recogntion model and
    #the secodn being the input to our focus detection model
    return (np.array(frames).reshape(10, 48, 48, 1).astype("float32"), open_face.astype("float32"))


#Because the DAiSEE dataset is very big, it wasn't practical to load the entire dataset all of once into our program
#Instead we created a generator that would process the videos and create the inputs as needed
class FrameGenerator:

    #constructor takes in the type of dataset (train, val, test)
    def __init__(self, dataset):
        self.dataset = dataset
    
    #everytime the dataset is called, it retrieves a single processed input for the model
    def __call__(self):
        
        #we get the path to the dataset
        if self.dataset == "train":
            data = train
            subset = "DataSet\Train"
            
        elif self.dataset == "val":
            data = val
            subset = "DataSet\Validation"
        else:
            data = test
            subset = "DataSet\Test"

        #if we're using a training set then we shuffle the dataset
        if self.dataset == "train":
            data = data.sample(frac = 1).reset_index(drop = True)

        fnames = data["ClipID"]

        #labels = np.asarray(data["Engagement"])

        labels = np.asarray(data["Boredom"])
        labels = labels.reshape( (1, len(labels)))

        #for each video file in the dataset
        for idx in range(len(fnames)):
            #we get the path
            path = os.path.join(path_prefix, subset)
            path = os.path.join(path, fnames.iloc[idx][:6])
            path = os.path.join(path, fnames.iloc[idx][:-4])
            path = os.path.join(path, fnames.iloc[idx])
            #and then yield the corresponding inputs to the model and their labels
            yield load_video(path), np.asarray(labels[:, idx])
        

output_signature = ( (tf.TensorSpec(shape = (10, 48, 48, 1), dtype = tf.float32),
                      tf.TensorSpec(shape = (10, 35), dtype = tf.float32)),
                    tf.TensorSpec(shape = (1), dtype = tf.int16))


train_ds = tf.data.Dataset.from_generator(FrameGenerator("train"),
                                          output_signature = output_signature)

val_ds = tf.data.Dataset.from_generator(FrameGenerator("val"),
                                        output_signature = output_signature)

test_ds = tf.data.Dataset.from_generator(FrameGenerator("test"),
                                        output_signature = output_signature)

BATCH_SIZE = 32
train_ds = train_ds.batch(BATCH_SIZE)

#we need to define the custom attention class in order to load in the model weights for the emotion
#recognition model
class Custom_Attention(tf.keras.layers.Layer):

        def call(self, x):
            size = tf.shape(x)
            x = tf.reshape(x, [size[0], size[1], 1])
            y = tf.einsum("...ij,...jk->...ik", x, tf.transpose(x, perm = [0, 2, 1]))/16.0
            y = tf.matmul(tf.nn.softmax(y), x)
            return tf.reshape(y, [size[0], size[1]])

#we implement our focus and emotion classifiers as layers in our model
class Emotion_Classifier(Layer):
    def __init__(self, **kwargs):
        super(Emotion_Classifier, self).__init__(**kwargs)

        emoti_model = Patt_Lite().model
        tf.keras.utils.get_custom_objects()["Custom_Attention"] = Custom_Attention
        emoti_model.load_weights("../../Models/Emotion_Rec/PAtt_Lite_weights.h5")

        inputs = keras.Input(shape = (48, 48, 1))
        
        y = emoti_model(inputs)
        
        #we attach a dense layer to our emotion classifier so that our emotion classifier can be trained
        #to detect engagement
        y = keras.layers.Dense(3, activation = "relu")(y)

        self.model = Model(inputs, y)

    def __call__(self, x):
        return self.model(x)

#for our focus classifier, we implemented it as an MLP. Since we're already working with high level features,
#our classifier doesn't need to be as big as the emotion classifier
class Focus_Classifier(Layer):
    def __init__(self, **kwargs):
        super(Focus_Classifier, self).__init__(**kwargs)
        inputs = keras.Input(shape = (35))
        y = keras.layers.Dense(64, activation = "relu")(inputs)
        y = keras.layers.Dense(64, activation = "relu")(y)
        y = keras.layers.Dense(128, activation = "relu")(y)
        y = keras.layers.Dense(3, activation = "relu")(y)
        self.model = Model(inputs, y)

    def __call__(self, x):
        return self.model(x)


#Finally our aggregation layer goes through each of the 10 sampled / averaged frames
#and evaluates the focus and emotion classifiers on each of them
class AggregationLayer(Layer):

    def __init__(self, emo_model, open_model, num_frames, **kwargs):
        super(AggregationLayer, self).__init__(**kwargs)
        self.emoti_model = emo_model
        self.open_model = open_model
        self.num_frames = num_frames
    
    def __call__(self, inputs):

        #we evaluate our two classifiers on each frame
        emot_outputs = [self.emoti_model(frame) for frame in tf.unstack(inputs[0], axis=1)]
        focus_outputs = [self.open_model(frame) for frame in tf.unstack(inputs[1], axis=1)]

        #average the outputs
        aver_emot_output = tf.reduce_mean(tf.stack(emot_outputs), axis=0)
        aver_focus_output = tf.reduce_mean(tf.stack(focus_outputs), axis=0)

        #and then our output from the aggregation layer are the two 3x1 averaged outputs
        aggregate_output = tf.concat([aver_emot_output, aver_focus_output], axis = 1)
        return aggregate_output

#our model takes in two inputs, the cropped image for the emotion classifier and the 
#10x35 array of features for our focus classifier
    
inp_emo = keras.Input((10, 48, 48, 1))
inp_open = keras.Input((10, 35))


emoti_model = Emotion_Classifier()
focus_model = Focus_Classifier()


y = AggregationLayer(emoti_model, focus_model, 10)([inp_emo, inp_open])

#we add a dense layer after our aggregation model so that the outputs from the emotion and focus classifiers
#are combined together
outputs = Dense(3, activation = "softmax")(y)


model = Model([inp_emo, inp_open], outputs)

model.summary()

model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer = keras.optimizers.Adam(learning_rate = 1e-3),
              metrics = ["acc"])

model.fit(train_ds,
          validation_data = val_ds,
          epochs = 10,
          batch_size = 32
          )
