from moviepy.editor import VideoFileClip
from time import sleep
import os
import pandas as pd
import csv
import sys
sys.path.append(os.environ['GARS_PROJ'])
from util import *

#this script splits each of the emotiw videos and converts them into 10 second clips

#each of these clips will be assumed to be representative of the entire video that they 
#came from and will be given the same label as the entire video. If necessary, we can
#try increasing the length of each clip to maybe 15-20 seconds

df = pd.read_excel(os.path.join(var.GARS_PROJ, "datasets", "EmotiW", "Engagement_Labels_Engagement.xlsx"), names = ["Filename", "Scores"], header = None)

df = df.set_index("Filename").T.to_dict("list")

#labels of each split clip will be stored here
scores = dict()

#This function splits a single video and will split it into 10 second clips.
#The clips will be stored in a folder called Split in the directory where the
#source video is stored
def split_video(fname, dataset):
    global scores
    #get the relative path to the source video
    set_path = os.path.join(var.GARS_PROJ, "datasets", "EmotiW", dataset)
    source_path = os.path.join(set_path, fname)
    
    current_duration = VideoFileClip(source_path).duration
    single_duration = 10
    count = int(current_duration/10)

    #we then split the video into 10 second segments
    while current_duration > single_duration:
        clip = VideoFileClip(source_path).subclip(current_duration-single_duration, current_duration)
        current_duration -= single_duration
        split_set_path = os.path.join(set_path, "Split")
        path = os.path.join(split_set_path, fname[:-4] + "_" + str(count) + ".mp4")

        #and place in the location described above
        clip.to_videofile(path, codec="libx264", temp_audiofile='temp-audio.m4a', remove_temp=True, audio_codec='aac')
        
        #we then assign the 10 second clip the same engagement as the one given to the 
        #entire video
        scores[path] = [path, df[fname[:-4]][0], dataset]
        count -= 1
        clip.reader.close()
        clip.audio.reader.close_proc()
        print("-----------------###-----------------")

#in this function we perform the split on each of the videos
#in a given directory
def extract_dataset(set):

    set_path = os.path.join(var.GARS_PROJ, "datasets", "EmotiW", set)
    for fname in os.listdir(set_path):
        if os.path.isfile(os.path.join(set_path, fname)):
            split_video(fname,set)
    


extract_dataset("Train")
extract_dataset("Validation")

col_names = ["Path", "Score", "Dataset"]

#and then we write the labels to a csv file
with open(os.path.join(var.GARS_PROJ, "datasets", "EmotiW", "Engagement_Labels_Split.csv"), "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(col_names)
    writer.writerows(scores.values())