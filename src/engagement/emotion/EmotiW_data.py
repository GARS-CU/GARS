import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import sys
sys.path.append(os.environ['GARS_PROJ'])
from util import *

############################ EmotiW Preprocessing Script ############################
#generates .npy files for the train and test set need to have the
#OpenFace_features folder and Engagement_Labels_Engagement.xlsx file in the same directory as this script

#we read in the file and convert to a dictionary

scores = pd.read_excel(os.path.join(var.GARS_PROJ, "datasets", "EmotiW", "Engagement_Labels_Engagement.xlsx"), names = ["Filename", "Scores"], header = None)

scores = scores.set_index("Filename").T.to_dict("list")

#reads in the all of the files located in the given directory
#into a numpy array. Since each video is around 6 minutes long, I broke
#them up into 10 second snippets and gave every snippet the same engagement label
#as the one given to the entire video (so this assumes that 10 seconds is enough
#to determine how engaged the user was for the entire 6 minutes, we can increase to 20
#or 30 seconds if necessary). In order to make the most of the dataset, I averaged over
#all of the OpenFace features captured for each frame within a single second. It looks like
# they were grabbing the features at a rate of 10 frames per second so for each second 
# I averaged the features taken over the 10 frames. It might be better to instead to take in 
#all of the frames in the dataset so that we have {frame0, frame10, ....}, {frame1, frame11, ....},
#{frame2, frame12,...} as data points. This would give us a lot more data (10x increase our training set
#would have 64000 samples instead) but a lot of it might end up being redundant and possibly noisy.

def generate_data(src_path):

    x = np.zeros((100*148, 10, 429))
    y = [0 for i in range(100*148)]

    index = 0
    #for each file in the directory
    for filename in tqdm(os.listdir(src_path)[:-1]):
        
        #we grab the engagement score
        if filename[-4:] == ".txt":
            score = float(scores[filename[:-4]][0])
        else:
            score = float(scores[filename][0])
        
        #we get the path to the file
        filepath = os.path.join(src_path, filename)

        #there was a weird issue with the dataset where all of the open face features were
        #comma separated but each row were surrounded by quotation marks so numpy and pandas
        #treated each row as a single entry
        with open(filepath, "r+") as f:
            txt = f.read().replace('"', "")
            f.seek(0)
            f.write(txt)
            f.truncate()

        #we generate the numpy array from the text file
        data = np.genfromtxt(filepath, invalid_raise = False, delimiter = ", ", skip_header = 1)

        #first two columns are frame # and time stamp so we remove those
        data = data[:,2:]

        #not all of the videos are of the same length so we split up the dataset into segments that are
        #each 100 frames long (each data point uses 100 frames) and remove what's left over
        data = data[1:100*(len(data)//100) + 1]

        #it doesn't happen that often but some of the frames have missing feature values
        # for those I just replaced them with the column wise average. I tried to just ignore those
        #particular values when averaging over every 10 frames but it looks like some of the features
        #were missing for entire seconds so I still had nan values
        col_mean = np.nanmean(data, axis = 0)
        indices = np.where(np.isnan(data))

        data[indices] = np.take(col_mean, indices[1])

        #for each frame there are 429 features, we're going to average over every 10 frames 
        try:
            data = data.reshape(data.shape[0]//10, 10, 429).mean(axis = 1)
        except:
            #There was one file that only had 321 features for a lot of the entries so I decided to skip it
            #I could have gone through the file and took out the rows with less than 429 entries but 
            #it was a single file and it would only be around 60 data points
            print("Error Gathering Data From " + filename + "\n")
            continue
        
        #each row in our data array represents the averaged features collected for a single second.
        #A single training point in our dataset corresponds to 10 seconds so we reshape the data array
        #so that it's in groups of 10
        x[index:index+data.shape[0]//10] =  data.reshape(data.shape[0]//10, 10, 429)

        #and we use the labels for the entire video for each of these 10 second snippets
        y[index:index + data.shape[0]//10] = [score for i in range(data.shape[0]//10)]
        index += data.shape[0]//10

    return x[:index], y[:index]

val_path = os.path.join(var.GARS_PROJ, "datasets", "EmotiW", "OpenFace_Features", "validation")
train_path = os.path.join(var.GARS_PROJ, "datasets", "EmotiW", "OpenFace_Features", "Train")

#we generate the train and test set
x_train, y_train = generate_data(train_path)
x_test, y_test=  generate_data(val_path)

#normalize the features
x_train = x_train.reshape(x_train.shape[0]*10, 429)
mean = x_train.mean(axis = 0)
std = x_train.std(axis = 0)
x_train = (x_train - mean)/std

#I was going to do PCA here but I thought it might be better to do it in a separate script
#pca = PCA(n_components = 50)
#x_train = pca.fit_transform(x_train).reshape((x_train.shape[0]//10, 10, 50))
x_train = x_train.reshape(x_train.shape[0]//10, 10, 429)
y_train = np.array(y_train)




x_test = x_test.reshape(x_test.shape[0]*10, 429)
x_test = (x_test - mean)/std
#x_test = pca.transform(x_test).reshape((x_test.shape[0]//10, 10, 50))
x_test = x_test.reshape(x_test.shape[0]//10, 10, 429)
y_test = np.array(y_test)

#export the numpy files

np.save(os.path.join(var.GARS_PROJ, "datasets", "EmotiW", "EmotiW_x_train.npy"), x_train)
np.save(os.path.join(var.GARS_PROJ, "datasets", "EmotiW", "EmotiW_y_train.npy"), y_train)
np.save(os.path.join(var.GARS_PROJ, "datasets", "EmotiW", "EmotiW_x_test.npy"), x_test)
np.save(os.path.join(var.GARS_PROJ, "datasets", "EmotiW", "EmotiW_y_test.npy"), y_test)
