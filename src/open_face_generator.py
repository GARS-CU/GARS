import os
import numpy as np
import tqdm

def gen_features(video_path):

    #we run OpenFace's FeatureExtraction executable which extracts the features for each frame in the video and exports them to a csv file.
    #I tried to see if there way to extract the features for one frame every second but I couldn't find a good way to do that.
    #In addition to the csv file, it also creates a video file which isn't really necessary either. It ends up taking around 6 seconds on my computer to 
    #extract the features for a 10 second video which is pretty intensive.

    #Right now, it only extracts the 35 action units but we could do things like head pose and eye gaze instead
    os.system(".\..\..\Models\OpenFace_2.2.0_win_x64\FeatureExtraction > $null -f " + video_path + " -aus -pose -gaze")

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
    data = data.reshape(10, 30, 329).mean(axis = 1)

    #Finally, we remove the folder containing the csv file
    os.system("rd /s /q processed")

    return data
def export(dataset, output_file):
    datapath = "..\..\datasets\DAiSEE\DataSet"
    with open(os.path.join(datapath, dataset + ".txt"), "r") as f:
        video_files = f.read().splitlines()
    
    print("Processing " + dataset + " Dataset")
    with open(output_file, "a") as out_file:
        for file in tqdm.tqdm(video_files):
            video_path = os.path.join(datapath, "Train")#dataset)
            video_path = os.path.join(video_path, file[:6])
            video_path = os.path.join(video_path, file[:-4])
            video_path = os.path.join(video_path, file)

            data = gen_features(video_path)
            np.savetxt(out_file, data, delimiter = ",")


export("Train", "..\..\datasets\DAiSEE\DataSet\Train_OpenFace.csv")
export("Validation", "..\..\datasets\DAiSEE\DataSet\Validation_OpenFace.csv")
export("Test", "..\..\datasets\DAiSEE\DataSet\Test_OpenFace.csv")