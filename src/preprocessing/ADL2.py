import os
import re
import pandas as pd
import math
import numpy as np
import csv

from datetime import time
from datetime import datetime 

# modify these file paths depending on what you are doing!!!!!!
# This scripts works on raw mmWave data to create seperate samples based on new activities with clock sync (SHOUT OUT TO NATHAN NG)
# Raw_Data_Dir = '/srv/scratch/mmwave/'
Path_To_GT = 'Participants/GroundTruth_Participants/GT_ISO/'
Path_To_Data = 'Participants/'
mmWave_Folder = "Radar/"

Raw_Data_Dir = 'Data_Raw/Data Collection/'
# Path_To_GT = 'Participants/GroundTruth_Participants/'

time_secs = [3600,60, 1]
FPS = 20
INDEX_TIME = 1
# INDEX_TIME = 0
RADAR_INDEX_TIME = 0
INDEX_ACTIVITY = 3
# INDEX_ACTIVITY = 2
INDEX_FRAME = 1
RADAR_FRAME_NUMBER = 1

TARGET_FRAME_NUMBER = 1
TARGET_ID_POINT_INDEX = 2
TARGET_ID_VALUE = 3

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
            print(gt_folder)
            process_camera_data(gt_folder, subject_file, 'Over')
            process_camera_data(gt_folder, subject_file, 'Side')

# this function will go through a specific orientation of data
# and link the subject CSV files to the assosciated ground truth file
# it will then check if there is no data missing, and if this condition is met, it will
# create processed data from the ground truth.


def process_camera_data(gt_folder, subject_file, orientation):
    # get the list of orientation folders inside the ground truth file
    gt_orientations = os.listdir(Raw_Data_Dir + Path_To_GT + gt_folder)
#     print(gt_orientations)
    directory = ""
    # if we match the folder orientation name with the orientation we pass in, we use that directories data.
    for gt_dir in gt_orientations:
        if(re.search(r""+orientation + "$", gt_dir, re.IGNORECASE) != None):
            directory = gt_dir
            break
    # now we need to find the matching subject folder with ground truth folder
    # if the subject folder is missing NO FILES (the 3 csvs needed for all subject folders)
    # then processing logic will be conducted to create action and transition data
    if directory != "":
        for gt_file in os.listdir(Raw_Data_Dir + Path_To_GT + gt_folder + "/" + directory):
            csv_files = get_CSV_files_from_gt(gt_file, subject_file, orientation)
            create_processed_data(Raw_Data_Dir + Path_To_GT + gt_folder +
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
    df = pd.read_csv(gt_file)
    frame_activity_pair = []
    frames = df.to_numpy()
    if(len(frames) == 0 or pd.isnull(frames[0][INDEX_TIME])):
        return
    experiment_start = 0
    start_time = 0
    print(gt_file)
    for i, frame in enumerate(frames):
        if(pd.isnull(frame[INDEX_TIME])):
            break
        if i == 0:
#             The start time of the ground truth. This is necessary because the mmwave and IR recordings start BEFORE
#             the GT. The start times must be aligned before we start assigning frames.
            experiment_start = frame[INDEX_TIME]
            curr_seconds = [float(numbers)
                         for numbers in frames[i][INDEX_TIME].split(':')]
            curr_seconds = sum([a*b for a, b in zip(time_secs, curr_seconds)])
            start_time = curr_seconds
        if(frames[i][INDEX_ACTIVITY] == "EndTime"):
            break
        time_recorded = [float(numbers)
                         for numbers in frames[i+1][INDEX_TIME].split(':')]
        seconds_passed = sum([a*b for a, b in zip(time_secs, time_recorded)])
        frame_activity_pair.append(
            [round(FPS*(seconds_passed - start_time)), frame[INDEX_ACTIVITY]])
        start_time = seconds_passed
    return frame_activity_pair, experiment_start

# this function will return the list of csv files from the subject folder assosciated with the ground truth files supplied
# this is done purely using regex, and the NAME of the csv files are stored in the list, not the actual csv file data.
# to find the specific files, the date stamped on the files are used to perform this matching. It is worth noting an empty
# list return represents that there is a missing subject folder for the assosciated ground truth data


def get_CSV_files_from_gt(gt_file, subject_file, orientation):
    file_date = re.search(r"-(\d+-\d+-\d+)_", gt_file).group(1)
    mmwave_directory = Raw_Data_Dir+Path_To_Data + \
        subject_file+"/"+orientation + "/"
    csv_files = []
    if(os.path.exists(Raw_Data_Dir+Path_To_Data+subject_file+"/"+orientation)):
        for mmwave_file in os.listdir(mmwave_directory + "Radar/"):
            if re.search(r"-" + file_date + "_", mmwave_file):
                for mmWave_csv in os.listdir(mmwave_directory + "Radar/" +mmwave_file):
#                     To avoid wasting space, we only care about the point clouds
                    # if mmWave_csv == "points_cloud.csv":
                    csv_files.append(mmwave_directory +
                        "Radar/" + mmwave_file+ "/" + mmWave_csv)
                return csv_files
    return csv_files


###############################################################################################################################
#                                                                                                                             #
#                                                                                                                             #
#                                  creating the processed data split based on actions                                         #
#                                                                                                                             #
#                                                                                                                             #
###############################################################################################################################

# the logic for creating action data is identical to the first half of create_transition_data
# however instead of storing it in an array, we just write it directly to the csv file.

def create_processed_data(gt_file, csv_files, orientation, subject_file):
    print("hi", gt_file)
    frame_activity_pair, experiment_start = process_gt_file(gt_file)
    print(experiment_start)
    experiment_start = time.fromisoformat(experiment_start)
    for csv_file in csv_files:
        print("-------- working on " + csv_file + " --------")
        count = 0   
        csv_component = re.search(r"\/([\w-]+\.csv)$", csv_file).group(1)
        experiment_component = csv_file.split("/")[6]
        experiment_component = experiment_component.split("_")[0]
        df = pd.read_csv(csv_file)
        # read the entire csv, convert into map with point for each frame
        # read the target_list csv and keep a map of which points to remove from the frames
        # after you have updated the map, flatten it into new df array
        headers = df.columns
        df = df.to_numpy()
        
        
        if csv_component == "points_cloud.csv":
            frames_with_noise = {}
            points_to_remove = {}
            id_file = csv_file.split("/")[0:7]
            id_file_path = '/'.join(id_file) + "/track_ids.csv"
            id_df = pd.read_csv(id_file_path)
            id_df = id_df.to_numpy()
            
            for id in id_df:
                # remember point is associated to PREVIOUS FRAME
                # print(id[TARGET_ID_POINT_INDEX], "----", id[TARGET_ID_VALUE])
                if math.isnan(float(id[TARGET_ID_POINT_INDEX])) or math.isnan(float(id[TARGET_ID_VALUE])):
                    continue
                else:
                    if id[TARGET_ID_VALUE] == 253 or id[TARGET_ID_VALUE] == 254 or id[TARGET_ID_VALUE] == 255:
                        # remove these frames
                        prev_frame = id[TARGET_FRAME_NUMBER] - 1
                        if prev_frame in points_to_remove:
                            points_to_remove[prev_frame].append(id[TARGET_ID_POINT_INDEX])
                        else:
                            points_to_remove[prev_frame] = []
                            points_to_remove[prev_frame].append(id[TARGET_ID_POINT_INDEX])
            
            for point in df:
                current_frame = point[RADAR_FRAME_NUMBER]
                if current_frame in frames_with_noise:
                    frames_with_noise[current_frame].append(point)
                else:
                    frames_with_noise[current_frame] = []
                    frames_with_noise[current_frame].append(point)
                    
            df = []
            # now that we got our points and hwich index to remove, we can just remove them manually
            for frame in points_to_remove.keys():
                # go through each point 1 by 1
                if not frame in frames_with_noise:
                    continue
                for point in points_to_remove[frame][::-1]:
                    # given frame and point, go to current frame map frame KEY and remove that point INDEX
                    if int(point) >= 0 and int(point) < len(frames_with_noise[frame]): 
                        del frames_with_noise[frame][int(point)]
                
                for point in frames_with_noise[frame]:
                    df.append(point)
                    
            df = np.array(df)
            # remake the df now
            
            
            
            
        ######################################################################################
        last_frame_added = 0
        index = 0
        for frame in df:
#             Certain parts of the radar dataset randomly have another header set in the middle.
#             This condition dodges them to prevent them from breaking everything.
            if frame[RADAR_INDEX_TIME] == "timestamp":
                index += 1
                continue
            isodate = datetime.fromisoformat(frame[RADAR_INDEX_TIME])
#             Find the index at which the time for the beginning of the GT is reached, so we can start assigning frames
            if isodate.time() >= experiment_start:
                print("broken at: " + str(index)+ "frame no.: " + str(frame[INDEX_FRAME]))
                break
            index += 1
        for activity_pair in frame_activity_pair:
            new_csv = "Data_Input/" + activity_pair[1] + "/" + orientation + \
                "/" + experiment_component + "_Sample_" + \
                str(count) + "/" + csv_component
            directory = re.split(r"[\w-]+\.csv$", new_csv)
#             print( directory[0])
            if(os.path.exists(directory[0]) == False):
                os.makedirs(directory[0])
            count += 1
            frames_to_add = activity_pair[0]
#             Frame warnings for IR, since the low frequency makes it likely 0 or 1 frames will coincide with the timing
            if frames_to_add == 1:
                print("1 warning" + experiment_component + str(count) + activity_pair[1])
            elif frames_to_add == 0:
                print("0 warning")
            frames_written = 0

            # load in to a map so you can see remove points properly
            

            with open(new_csv, 'w+', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(headers)
                
                # first_frame = df[index][INDEX_FRAME]
                for row in df[index:]:
#                     This check is done first in case the frames_to_add is 0 (especially for IR data where FPS is 1)
                    if(last_frame_added != row[INDEX_FRAME]):
                        last_frame_added = row[INDEX_FRAME]
                        frames_written += 1
                    if frames_written == frames_to_add + 1:
                        break
                    writer.writerow(row)
                    index += 1             
                    
process_mmWave_data()