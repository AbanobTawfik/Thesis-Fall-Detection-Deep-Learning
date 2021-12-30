import os
import re
import pandas as pd
import datetime
import time
import numpy as np
import csv

# modify these file paths depending on what you are doing!!!!!!
# This scripts works on Ariyamehr's data to create seperate samples based on new activities (ALL WRITTEN BY ABANOB TAWFIK)
Raw_Data_Dir = 'E:/dataset/'
output_csv_dir = 'E:/dataset/Data_Input/' 
count = 0

# since the ground truth and subject folder both share the same naming ID and are all unique to each other
# we are able to use regex to match the pattern to link the correct ground truth folder to the subject file based on ID
# in this case, the id is just the number next to the name for the subject file

def process_file(file, count):
    df = pd.read_csv(file)
    headers = df.columns
    df = df.to_numpy()
    headers = headers[0:10]
    # we want to make a new sample for ADL/FALLING everytime the activity changes
    activity = df[0][12]
    rows = []
    for row in df:
        if row[12] != activity:
            new_csv = output_csv_dir + activity + "/sample" + str(count) + ".csv"
            print(new_csv)
            count = count + 1
            with open(new_csv, 'w+', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(headers)
                for row2 in rows:
                    writer.writerow(row2)
                    
            rows = []
            activity = row[12]
        else:
            if not (row[11] == -1 or row[11] == 253 or row[11] == 254 or row[11] == 255):
                rows.append(row[0:10])
    return count
    
def process():
    count = 0
    for file in os.listdir(Raw_Data_Dir):
        if file == "Over" or file == "Side":
            for csv_dir in os.listdir(Raw_Data_Dir + file):
                for csv_files in os.listdir(Raw_Data_Dir + file + "/" + csv_dir):
                    if (csv_files == "points_cloud_clean_target_ids_labelled_cut.csv"):
                            count = process_file(Raw_Data_Dir + file + "/" + csv_dir + "/" + csv_files, count)
                            
process()