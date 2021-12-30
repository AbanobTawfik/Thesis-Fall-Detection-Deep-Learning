import os
import re
import pandas as pd
import datetime
import time
import numpy as np
import csv

# modify these file paths depending on what you are doing!!!!!!
# This scripts works on raw mmWave data to create seperate samples based on new activities with no clock sync (ALL WRITTEN BY ABANOB TAWFIK)
# comment block for scratch
# uncomment block for local
Raw_Data_Dir = 'Data_Raw/Data Collection/'
Path_To_GT = 'Participants/GroundTruth_Participants/'
Path_To_Data = 'Participants/'
mmWave_Folder = "Radar/"
onedrive_prefix = '~/OneDrive/data_analysis_mmwave/'

# comment block for local
# uncomment block for scratch
# Raw_Data_Dir = '/srv/scratch/mmwave/'
# Path_To_GT = 'Participants/GroundTruth_Participants/'
# Path_To_Data = 'Participants/'
# mmWave_Folder = ""
# onedrive_prefix = '/srv/scratch/mmwave/all_data/'

time_secs = [60, 1]
FPS = 20
TRANSITION_TIME = 2
INDEX_TIME = 0
INDEX_ACTIVITY = 2
INDEX_FRAME = 1
INDEX_ACTIVITY_NAME = 1

# since the ground truth and subject folder both share the same naming ID and are all unique to each other
# we are able to use regex to match the pattern to link the correct ground truth folder to the subject file based on ID
# in this case, the id is just the number next to the name for the subject file


def process_mmWave_data():
    for subject_file in os.listdir(Raw_Data_Dir + Path_To_Data):
        # stripping the number from the file name
        subject_number = re.search('Subject-(\d+)_', subject_file)
        if subject_number != None:
            subject_number = subject_number.group(1)
            gt_folder = find_gt_for_subject_number(subject_number)
            process_camera_data(gt_folder, subject_file, 'Over')
            process_camera_data(gt_folder, subject_file, 'Side')

# this function will go through a specific orientation of data
# and link the subject CSV files to the assosciated ground truth file
# it will then check if there is no data missing, and if this condition is met, it will
# create processed data from the ground truth.


def process_camera_data(gt_folder, subject_file, orientation):
    # get the list of orientation folders inside the ground truth file
    gt_orientations = os.listdir(Raw_Data_Dir + Path_To_GT + gt_folder)
    directory = ""
    # if we match the folder orientation name with the orientation we pass in, we use that directories data.
    for gt_dir in gt_orientations:
        if(re.search(r""+orientation + "$", gt_dir, re.IGNORECASE) != None):
            directory = gt_dir
            break
    # now we need to find the matching subject folder with ground truth folder
    # if the subject folder is missing NO FILES (the 3 csvs needed for all subject folders)
    # then processing logic will be conducted to create action and transition data
    for gt_file in os.listdir(Raw_Data_Dir + Path_To_GT + gt_folder + "/" + directory):
        csv_files = get_CSV_files_from_gt(gt_file, subject_file, orientation)
        if len(csv_files) == 3:
            create_processed_data(Raw_Data_Dir + Path_To_GT + gt_folder +
                                  "/" + directory + "/" + gt_file, csv_files, orientation, subject_file)
            create_transition_data(Raw_Data_Dir + Path_To_GT + gt_folder +
                                   "/" + directory + "/" + gt_file, csv_files, orientation, subject_file)


###############################################################################################################################
#                                                                                                                             #
#                                                                                                                             #
#                                  helper functions for linking ground truth files to raw data                                #
#                                                                                                                             #
#                                                                                                                             #
###############################################################################################################################

# this function will return the ground truth file assosciated with the
# linked subject file based on the number using regex.
# this can be done since the naming scheme is unique


def find_gt_for_subject_number(number):
    for gt_file in os.listdir(Raw_Data_Dir + Path_To_GT):
        if(re.search(r"" + number + "$", gt_file) != None):
            return gt_file

# this function will simply read through a ground truth excel file
# and create a list of pairs that consist of
# 1. the activity being performed
# 2. how many frames this activity is performed for
# the frames are acquired by multiplying the given FPS of the radar with the number of
# seconds the action occurs for (rounded mathematically)


def process_gt_file(gt_file):
    if(gt_file == "Data_Raw/Data Collection/Participants/GroundTruth_Participants/GT_Labelled_05/GT-over/over-05-07-00_20200819-110253_GT.xlsx"):
        print("HIOYOOYOO")
    df = pd.read_excel(gt_file, engine='openpyxl')
    frame_activity_pair = []
    frames = df.to_numpy()
    if(len(frames) == 0 or pd.isnull(frames[0][INDEX_TIME])):
        return
    
    start_time = 0
    for i, frame in enumerate(frames):
        if(pd.isnull(frame[INDEX_TIME])):
            break
        if(frames[i][INDEX_ACTIVITY] == "EndTime"):
            break
        time_recorded = [float(numbers)
                         for numbers in frames[i+1][INDEX_TIME].split(':')]
        seconds_passed = sum([a*b for a, b in zip(time_secs, time_recorded)])
        frame_activity_pair.append(
            [round(FPS*(seconds_passed - start_time)), frame[INDEX_ACTIVITY]])
        start_time = seconds_passed
    return frame_activity_pair


# this function will return the list of csv files from the subject folder assosciated with the ground truth files supplied
# this is done purely using regex, and the NAME of the csv files are stored in the list, not the actual csv file data.
# to find the specific files, the date stamped on the files are used to perform this matching. It is worth noting an empty
# list return represents that there is a missing subject folder for the assosciated ground truth data


def get_CSV_files_from_gt(gt_file, subject_file, orientation):
    file_date = re.search(r"-(\d+-\d+-\d+)_", gt_file).group(1)
    mmwave_directory = Raw_Data_Dir+Path_To_Data + \
        subject_file+"/"+orientation + "/" + mmWave_Folder
    # print(Raw_Data_Dir+Path_To_Data+subject_file+"/"+orientation)
    csv_files = []
    if(os.path.exists(Raw_Data_Dir+Path_To_Data+subject_file+"/"+orientation)):
        for mmwave_file in os.listdir(mmwave_directory):
            if re.search(r"-" + file_date + "_", mmwave_file):
                for mmWave_csv in os.listdir(mmwave_directory + mmwave_file):
                    csv_files.append(mmwave_directory +
                                     mmwave_file+"/" + mmWave_csv)
                return csv_files
    return csv_files


###############################################################################################################################
#                                                                                                                             #
#                                                                                                                             #
#                                  creating the processed data split based on actions                                         #
#                                                                                                                             #
#                                                                                                                             #
###############################################################################################################################

# the logic for CREATING the new files remains the same for both processing acitons and transitions, however the key difference
# is in what data is being stored.

def create_transition_data(gt_file, csv_files, orientation, subject_file):
    frame_activity_pair = process_gt_file(gt_file)
    # if the action is not end signal we want to take the last 2 seconds (or all available if < 2 secs) of frames from previous action
    # add the last 2 seconds (or all available if < 2 secs) of frames from current action to file
    # save these under transitions
    for csv_file in csv_files:
        print("------------- working on transitions for  " +
              csv_file + " -------------")
        count = 0
        rows_added = 0
        csv_component = re.search(r"\/(\w+\.csv)$", csv_file).group(1)
        df = pd.read_csv(csv_file)
        headers = df.columns.to_numpy()
        last_frame_added = 0
        # need to create the array that contains 2x1 array consisting of the activity name and the assosciated
        # rows for all the frames (which is an array itself)
        all_activities = []
        for activity_pair in frame_activity_pair:
            activity_data = []
            frames_to_add = activity_pair[0]
            tmp = last_frame_added
            for frame in df.to_numpy()[rows_added:]:
                if(last_frame_added != frame[INDEX_FRAME]):
                    last_frame_added += 1
                if (last_frame_added - tmp - 1) == frames_to_add:
                    break
                activity_data.append(frame)
                rows_added += 1
            all_activities.append(
                [activity_pair[INDEX_ACTIVITY_NAME], activity_data])

        # creating transitions from all actions up until the end signal. Note we go from 0 -> n-1 since there is
        # no transitions from end signal -> nothing. This will be done using the array above
        for i in range(0, (len(all_activities) - 1)):
            new_csv = "Data_Input/Transitions/" + all_activities[i][0] + "_to_" + all_activities[i+1][0] + "/" + orientation + \
                "/" + subject_file + "_Sample_" + \
                str(count) + "/" + csv_component
            directory = re.split(r"\w+\.csv$", new_csv)
            if(os.path.exists(directory[0]) == False):
                os.makedirs(directory[0])
            if(os.path.exists(new_csv)):
                count += 1
            frames_to_add_from = FPS*TRANSITION_TIME
            frames_to_add_to = FPS*TRANSITION_TIME
            # to get the frames for our from transitions, we need to get them from the of the current activity
            # and traverse the array backwards until we have the amount of frames required.
            # to do this we can reverse the array using python indexing and take from the start of the array
            # simulating traversing backwards.
            transition_from = []
            last_frame_added_from = all_activities[i][1][::-1][0][INDEX_FRAME]
            for from_frame in all_activities[i][1][::-1]:
                if(last_frame_added_from != from_frame[INDEX_FRAME]):
                    frames_to_add_from -= 1
                    last_frame_added_from = from_frame[INDEX_FRAME]
                if(frames_to_add_from == 0):
                    break
                transition_from.append(from_frame)
            # to get the frames for our to transitions, we need to get them from the next activity from our current one
            # however this time we traverse the array from the start until we have the amount of frames required.
            transition_to = []
            last_frame_added_to = all_activities[i+1][1][0][INDEX_FRAME]
            for to_frame in all_activities[i+1][1]:
                if(last_frame_added_to != to_frame[INDEX_FRAME]):
                    frames_to_add_to -= 1
                    last_frame_added_to = to_frame[INDEX_FRAME]
                if(frames_to_add_to == 0):
                    break
                transition_to.append(to_frame)
            # now combine the to and from frame into 1 transition in the new csv file
            with open(new_csv, 'w+', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(headers)
                # just add the from first, then the to
                for frame in transition_from:
                    writer.writerow(frame)
                for frame in transition_to:
                    writer.writerow(frame)

# the logic for creating action data is identical to the first half of create_transition_data
# however instead of storing it in an array, we just write it directly to the csv file.

def create_processed_data(gt_file, csv_files, orientation, subject_file):
    frame_activity_pair = process_gt_file(gt_file)
    for csv_file in csv_files:
        print("-------- working on " + csv_file + " --------")
        count = 0
        rows_added = 0
        csv_component = re.search(r"\/(\w+\.csv)$", csv_file).group(1)
        df = pd.read_csv(csv_file)
        headers = df.columns.to_numpy()
        last_frame_added = 0
        for activity_pair in frame_activity_pair:
            new_csv = "Data_Input/" + activity_pair[1] + "/" + orientation + \
                "/" + subject_file + "_Sample_" + \
                str(count) + "/" + csv_component
            directory = re.split(r"\w+\.csv$", new_csv)
            if(os.path.exists(directory[0]) == False):
                os.makedirs(directory[0])
            if(os.path.exists(new_csv)):
                count += 1
            frames_to_add = activity_pair[0]
            tmp = last_frame_added
            with open(new_csv, 'w+', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(headers)
                for frame in df.to_numpy()[rows_added:]:
                    if(last_frame_added != frame[INDEX_FRAME]):
                        last_frame_added += 1
                    if (last_frame_added - tmp - 1) == frames_to_add:
                        break
                    writer.writerow(frame)
                    rows_added += 1


process_mmWave_data()
