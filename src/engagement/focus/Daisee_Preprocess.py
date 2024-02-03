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
import tqdm
import sys
sys.path.append(os.environ['GARS_PROJ'])
from util import *
#This program implements and trains the complete engagement detection model (emotion + focus classifier) on the DAiSEE dataset

#get the train, val, and test dataframes containing the paths to the video files in each dataset as well as the labels

path_prefix = os.path.abspath(os.path.join(var.GARS_PROJ, "datasets", "DAiSEE"))

train = pd.read_csv(os.path.join(path_prefix, "Labels", "TrainLabels.csv"))
test = pd.read_csv(os.path.join(path_prefix, "Labels", "TestLabels.csv"))
val = pd.read_csv(os.path.join(path_prefix, "Labels", "ValidationLabels.csv"))

#The Boredom labels were a lot more balanced than the Engagement labels so I thought it might be good to use them instead
#We can switch over to engagement though and make the dataset more balanced if needed

#There weren't that many videos that were classified as extremely bored so I lumped the two highest boredom scores together
#train.loc[train["Boredom"] == 3, "Boredom"] = 2



#we load in the haar cascade classifier used for face detection. This model has some trouble with detecting 
#faces when they're rotated so if it can't detect a face, I used a more intensive neural network from the deepface library to crop the face.
#The bigger model takes around 30 seconds to evaluate so I only used it if the haar cascade model couldn't find anything.
face_cascade = cv2.CascadeClassifier(f"{var.GARS_PROJ}/haarcascade_frontalface_alt.xml")


train_data_loc = dict()
val_data_loc = dict()

train_openface = open(os.path.join(var.GARS_PROJ, "datasets", "DAiSEE_openface_allfeatures", "Train_OpenFace.csv"), "r")
val_openface = open(os.path.join(var.GARS_PROJ, "datasets" , "DAiSEE_openface_allfeatures" ,"Validation_OpenFace.csv"), "r")
            
for index, line in enumerate(train_openface):
    if (index % 11) == 0:
        train_data_loc[line.strip()] = index

for index, line in enumerate(val_openface):
    if (index % 11) == 0:
        val_data_loc[line.strip()] = index


#This function index into a csv file to obtain the OpenFace features that we had extracted from each video prior to training
#Takes in the path to the video file. Returns a 10x329 numpy array with each row representing the averaged facial features
#for each second which is used as the input for our focus classifier
        
def gen_features(video_path):

    if "Train" in video_path:
        loc = train_data_loc[os.path.basename(video_path)[:-4]]
        file = train_openface
    else:
        loc = val_data_loc[os.path.basename(video_path)[:-4]]
        file = val_openface

    file.seek(0)

    return np.loadtxt(file, skiprows = loc + 1, max_rows = 10, delimiter = ",")


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
            processed_frame = cv2.resize(processed_frame,(100 , 100))

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
    return (np.array(frames).reshape(10, 100, 100, 1).astype("float32"), open_face.astype("float32"))


def gen_features(video_path):

    if "Train" in video_path:
        loc = train_data_loc[os.path.basename(video_path)[:-4]]
        file = train_openface
    else:
        loc = val_data_loc[os.path.basename(video_path)[:-4]]
        file = val_openface

    file.seek(0)

    return np.loadtxt(file, skiprows = loc + 1, max_rows = 10, delimiter = ",")


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
                        print("Unable to Detect Face!")
                        #and if that doesn't work then we give a blank image for that frame (this rarely happens)
                        processed_frame = np.zeros((100, 100))
            else:
                #if multiple faces are found, we take the largest one
                face_location = [max(face_locations, key = lambda x: x[2]*x[3])]

                for x,y,h,w in face_location:
                    #we then crop out the face
                    processed_frame = processed_frame[y:y+h, x:x+w]
            

            #we downsample the image to a 48x48 array
            processed_frame = cv2.resize(processed_frame,(100 ,100))

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
    return (np.array(frames).reshape(10, 100, 100, 1).astype("float32"), open_face.astype("float32"))




def export(dataset):
    if dataset == "train":
        data = train
        subset = os.path.join("DataSet", "Train")
            
    elif dataset == "val":
        data = val
        subset = os.path.join("DataSet", "Validation")
    else:
        data = test
        subset = os.path.join("DataSet", "Test")
    
    fnames = data["ClipID"]

    labels = np.asarray(data["Boredom"])
    
    open_features = np.zeros((len(labels), 10, 709))
    emotion_features = np.zeros((len(labels), 10, 100, 100, 1))

    dropped_indices = []
    for idx in tqdm.tqdm(range(len(labels))):
        path = os.path.join(path_prefix, subset)
        path = os.path.join(path, fnames.iloc[idx][:6])
        path = os.path.join(path, fnames.iloc[idx][:-4])
        path = os.path.join(path, fnames.iloc[idx])

        try:
            inp = load_video(path)
        except:
                print("Error removing " + path + " from list")
                dropped_indices.append(idx)
                continue
        emotion_features[idx] = inp[0]
        open_features[idx] = inp[1]
    
    labels = np.delete(labels, dropped_indices, axis = 0)
    emotion_features = np.delete(emotion_features, dropped_indices, axis = 0)
    open_features = np.delete(open_features, dropped_indices, axis = 0)

    np.save(os.path.join(path_prefix, subset, "emotion_large.npy"), emotion_features)
    np.save(os.path.join(path_prefix, subset, "open_large.npy"), open_features)

    np.save(os.path.join(path_prefix, subset, "labels.npy"), labels)


for data in ["val", "train"]:

    print("\n###################Exporting " + data + " dataset###################\n")
    export(data)
