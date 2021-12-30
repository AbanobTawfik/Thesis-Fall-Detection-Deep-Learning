from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as kb
from tensorflow.keras.layers import Input, TimeDistributed, Flatten, Lambda, Concatenate, Reshape, LSTM, RepeatVector, SimpleRNN, Activation, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers, models, optimizers, callbacks
from datetime import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import json
import os
import re
from matplotlib import pyplot
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
disable_eager_execution()

epochs = 5
ARRAY_CHUNKS = 1000
TIMESTAMP = 0
X = 2
Y = 3
Z = 4
DOPPLER = 5
MAX_POINT_FRAME = -1
STEP_SIZE = 1

# removing test set from input
ADL_Directories = os.listdir('Data_Input/')
ADL_Directories.remove('Falling')
# ADL_Directories.remove('Transitions')
Falling_Directories = ['Falling']
#onedrive_prefix = '~/OneDrive/data_analysis_mmwave/'
Array_DIR_Train = "Training_Data_Array/"
Array_DIR_Test = "Testing_Data_Array/"
Frame_DIR_Stats = "Frame_Statistics/"

if os.path.exists(Array_DIR_Train) is False:
    os.mkdir(Array_DIR_Train)
if os.path.exists(Frame_DIR_Stats) is False:
    os.mkdir(Frame_DIR_Stats)
if os.path.exists(Array_DIR_Test) is False:
    os.mkdir(Array_DIR_Test)
if os.path.exists("Checkpoints/") is False:
    os.mkdir("Checkpoints")

###############################################################################################################################
#                                                                                                                             #
#                                                                                                                             #
#                                             Processing, Saving and Loading Data                                             #
#                                                                                                                             #
#                                                                                                                             #
###############################################################################################################################


def train_HVRNNAE(training_set, testing_set):
    number_of_frames = 10
    number_of_points = 64
    number_of_features = 4

    encoding_dimension = 64
    latent_dimension = 16

    inputs = Input(
        shape=(number_of_frames, number_of_points, number_of_features))
    input_flatten = TimeDistributed(Flatten(None))(inputs)
    input_mean = TimeDistributed(
        Dense(encoding_dimension, activation=None), name='input_mean')(input_flatten)
    input_log_variance = TimeDistributed(
        Dense(encoding_dimension, activation=None), name='input_log_variance')(input_flatten)
    sampled_input = Lambda(sample)([input_mean, input_log_variance])

    encoder = SimpleRNN(latent_dimension, activation='tanh',
                        return_sequences=False)(sampled_input)
    repeat_encoder = RepeatVector(number_of_frames)(encoder)
    decoder_RNN = SimpleRNN(
        latent_dimension, activation='tanh', return_sequences=True)(repeat_encoder)
    decoder = Lambda(lambda x: tf.reverse(x, axis=[-2]))(decoder_RNN)

    latent_input = TimeDistributed(
        Dense(encoding_dimension, activation='tanh'))(decoder)
    latent_mean = TimeDistributed(
        Dense(number_of_features, activation=None))(latent_input)
    latent_log_variance = TimeDistributed(
        Dense(number_of_features, activation=None))(latent_input)

    output = Concatenate()([latent_mean, latent_log_variance])
    output = TimeDistributed(RepeatVector(number_of_points))(output)
    outputs = TimeDistributed(
        Reshape((number_of_points, number_of_features*2)), name='test')(output)

    def HVRNNAE_loss(y_t, y_p):
        batch_size = kb.shape(y_t)[0]
        number_of_frames = kb.shape(y_t)[1]
        number_of_features = kb.shape(y_t)[-1]

        predicted_mean = y_p[:, :, :, :number_of_features]
        predicted_log_variance = y_p[:, :, :, number_of_features:]
        predicted_variance = kb.exp(predicted_log_variance)

        true_reshape = kb.reshape(y_t, (batch_size, number_of_frames, -1))
        mean_reshape = kb.reshape(
            predicted_mean, (batch_size, number_of_frames, -1))
        variance_reshape = kb.reshape(
            predicted_variance, (batch_size, number_of_frames, -1))
        log_variance_reshape = kb.reshape(
            predicted_log_variance, (batch_size, number_of_frames, -1))

        log_output = (kb.square(true_reshape - mean_reshape))/variance_reshape
        log_output = kb.sum(0.5*log_output, axis=-1)

        KL_loss = -0.5*kb.sum(1 + input_log_variance -
                              kb.square(input_mean) - kb.exp(input_log_variance), axis=-1)
        return kb.mean(log_output + KL_loss)

    model = Model(inputs, outputs)
    optimiser = optimizers.Adam()
    model.compile(optimizer=optimiser, loss=HVRNNAE_loss)

    checkpoint_path = "Checkpoints/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint_callback = ModelCheckpoint(
        checkpoint_path, save_weights_only=True, verbose=1)

    last_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    checkpoint_number = re.search('cp-(\d+).ckpt', last_checkpoint)
    remaining_epochs = epochs
    if checkpoint_number != None:
        remaining_epochs = epochs - int(checkpoint_number.group(1))
        model.load_weights(last_checkpoint)
    print("Restored model from epoch ", str(checkpoint_number.group(1)),
          ", performing ", str(remaining_epochs), " epochs:")
    if remaining_epochs > 0:
        model.fit(training_set, training_set, epochs=remaining_epochs, batch_size=8, shuffle=False,
                  validation_data=(testing_set, testing_set), verbose=1, callbacks=[checkpoint_callback])
    print("model finished training")
    return model


def predict(model, inference_data):
    print("#################Predicting####################")
    kb.clear_session()
    output_mean = Model(inputs=model.input,
                        outputs=model.get_layer('input_mean').output)
    output_log_variance = Model(
        inputs=model.input, outputs=model.get_layer('input_log_variance').output)
    predictions = []
    losses = []
    for data in inference_data:
        data = np.expand_dims(data, axis=0)
        current_prediction = model.predict(data, batch_size=1)
        predicted_output_mean = output_mean.predict(data, batch_size=1)
        predicted_log_variance = output_log_variance.predict(
            data, batch_size=1)
        current_loss = loss(data, current_prediction,
                            predicted_output_mean, predicted_log_variance)
        losses.append(current_loss)
    return losses
    # model.add()


def loss(y_t, y_p, output_mean, output_log_variance):
    batch_size = kb.shape(y_t)[0]
    number_of_frames = kb.shape(y_t)[1]
    number_of_features = kb.shape(y_t)[-1]

    predicted_mean = y_p[:, :, :, :number_of_features]
    predicted_log_variance = y_p[:, :, :, number_of_features:]
    predicted_variance = kb.exp(predicted_log_variance)

    true_reshape = kb.reshape(y_t, (batch_size, number_of_frames, -1))
    mean_reshape = kb.reshape(
        predicted_mean, (batch_size, number_of_frames, -1))
    variance_reshape = kb.reshape(
        predicted_variance, (batch_size, number_of_frames, -1))
    log_variance_reshape = kb.reshape(
        predicted_log_variance, (batch_size, number_of_frames, -1))

    log_output = (kb.square(true_reshape - mean_reshape))/variance_reshape
    log_output = kb.sum(0.5*log_output, axis=-1)

    KL_loss = -0.5*kb.sum(1 + output_log_variance -
                          kb.square(output_mean) - kb.exp(output_log_variance), axis=-1)
    return kb.mean(log_output + KL_loss)


def sample(inputs):
    input_mean, input_log_variance = inputs
    batch_size = kb.shape(input_mean)[0]
    number_of_frames = kb.int_shape(input_mean)[1]
    latent_dimension = kb.int_shape(input_mean)[2]
    epsilon = kb.random_normal(shape=(
        batch_size, number_of_frames, latent_dimension), mean=0., stddev=1.0, seed=None)
    return input_mean + kb.exp(0.5*input_log_variance) * epsilon
# loss, accuracy = model.evaluate(test, test, verbose = 1)
# print("Model accuracy: {:5.2f}%".format(100*accuracy))
#output = predict(model, test)
# print(output)
# step 3 if detected a spike in loss function -> pass through the check to see if it is a fall by using
# the change in the centroid from the spike.


###############################################################################################################################
#                                                                                                                             #
#                                                                                                                             #
#                                                    helper functions for data pre-processing                                 #
#                                                                                                                             #
#                                                                                                                             #
###############################################################################################################################


# this method loads in the raw data from the input directory and returns a sequence of frames that capture motion history with
# motion is capture in sequences of 10 frames at a time and the window has a step size of 1
def load_raw_data_sliding_window(Directory):
    raw_data = []
    for activity in Directory:
        if activity == "Transitions":
            for transition in os.listdir('Data_Input/' + activity):
                for orientation in os.listdir('Data_Input/' + "/" + activity + "/" + transition):
                    for subject in os.listdir('Data_Input/' + activity + "/" + transition + "/" + orientation):
                        data = pd.read_csv(
                            'Data_Input/' + activity + "/" + transition + "/" + orientation + "/" + subject + "/points_cloud.csv")
                        raw_data.append(data.to_numpy())
        else:
            for orientation in os.listdir('Data_Input/' + activity):
                for subject in os.listdir('Data_Input/' + activity + "/" + orientation):
                    data = pd.read_csv('Data_Input/' + activity + "/" +
                                       orientation + "/" + subject + "/points_cloud.csv")
                    raw_data.append(data.to_numpy())
    return raw_data


def check_if_float_string(input):
    try:
        float(input)
        return True
    except ValueError:
        return False


def remove_junk(all_frames):
    clean_frames = []
    for frame in all_frames:
        add = True
        for point in frame:
            if not check_if_float_string(point[0]) or not check_if_float_string(point[1]) or not check_if_float_string(point[2]) or not check_if_float_string(point[3]):
                add = False
        if(add):
            clean_frames.append(frame)
    return clean_frames

# this method will process raw data, by oversampling points to a fixed dimension for each frame and proceed to
# save the data to disk. it will only process unprocessed data.


def oversample_and_save_remaining_frames(frame_sequences, saved_array_parts, Array_DIR, TrainTest):
    count = 0
    last_loaded_array_index = saved_array_parts
    all_frames = []
    for frame in frame_sequences.values():
        all_frames.append(frame)
    all_frames = remove_junk(all_frames)
    frame_sequence_array = []
    for i in range(0, len(all_frames) - 10, STEP_SIZE):
        frame_sequence_array.append(np.squeeze([all_frames[i:(i+10)]]))

    # if there is still data to process, start from last save point
    processed_oversampled_frames_array = []
    processed_oversampled_frames = []
    if (len(frame_sequence_array) - saved_array_parts*ARRAY_CHUNKS) > ARRAY_CHUNKS:
        for frame_batch in frame_sequence_array[saved_array_parts*10:]:
            if count == ARRAY_CHUNKS or (count == len(frame_sequence_array[saved_array_parts*ARRAY_CHUNKS:]) - 1):
                np.save(Array_DIR + TrainTest + str(last_loaded_array_index), np.array(
                    processed_oversampled_frames), allow_pickle=True, fix_imports=True)
                processed_oversampled_frames = []
                count = 0
                last_loaded_array_index += 1
            batch = []
            for frame in frame_batch:
                frame_sampled = oversample(frame)
                batch.append(frame_sampled)
            processed_oversampled_frames.append(batch)
            count += 1
# this method will load any data that is saved during raw data processing, this saves time
# making sure there is no duplicate pre-processing


def load_saved_array_data(saved_array_parts, Array_Dir, TrainTest):
    processed_oversampled_frames = []
    if saved_array_parts > 0:
        for i in range(saved_array_parts):
            array_part = np.load(Array_Dir + TrainTest + str(i) + ".npy")
            for value in array_part:
                processed_oversampled_frames.append(value)
    return np.array(processed_oversampled_frames)


def oversample(data):
    data = np.array(data, dtype=np.float64)
    if(len(data) == 0):
        return []
    number_of_points = MAX_POINT_FRAME
    axis = np.shape(data)[0]
    mean = np.mean(data, axis=0)
    frame_np = np.sqrt(number_of_points/axis)*data + \
        mean - np.sqrt(number_of_points/axis)*mean
    oversampled_frame = frame_np.tolist()
    oversampled_frame.extend([mean]*(number_of_points-axis))
    oversampled_return = np.array(oversampled_frame)
    return oversampled_return


def generate_frame_sequences(raw_data_samples):
    frame_map = {}
    timestamps = {}
    count = 0
    sample_string = "sample_" + str(count)
    for sample in raw_data_samples:
        for data in sample:
            timestamp = data[TIMESTAMP]+"_"+sample_string
            if timestamp in frame_map:
                frame_map[timestamp].append(np.array(data[[X, Y, Z, DOPPLER]]))
            else:
                frame_map[timestamp] = []
                frame_map[timestamp].append(np.array(data[[X, Y, Z, DOPPLER]]))
        count = count + 1
        sample_string = "sample_" + str(count)
    return frame_map


def load_mmData(Directory, TrainTest, Array_DIR):
    point_cloud_scatter_samples = load_raw_data_sliding_window(Directory)
    frame_sequences = generate_frame_sequences(point_cloud_scatter_samples)
    saved_array_parts = len(os.listdir(Array_DIR))
    oversample_and_save_remaining_frames(
        frame_sequences, saved_array_parts, Array_DIR, TrainTest)
    return load_saved_array_data(saved_array_parts, Array_DIR, TrainTest)


def compute_frame_statistics(train, test):
    max = -1
    min = 10000000
    average = 0
    histogram = {}
    count = 0
    for frame in train.keys():
        histogram[frame] = len(train[frame])
        average = average + len(train[frame])
        count = count + 1
        if(len(train[frame]) > max):
            max = len(train[frame])
        if(len(train[frame]) < min):
            min = len(train[frame])

    for frame in test.keys():
        histogram[frame] = len(test[frame])
        average = average + len(test[frame])
        count = count + 1
        if(len(test[frame]) > max):
            max = len(test[frame])
        if(len(test[frame]) < min):
            min = len(test[frame])
    with open(Frame_DIR_Stats + 'Frame_Histogram.json', 'w') as file:
        json.dump(histogram, file)
    with open(Frame_DIR_Stats + 'max_frame_size.json', 'w') as file:
        json.dump({"max": max, "min": min, "average": average/count}, file)


def print_histogram():
    histogram = {}
    with open(Frame_DIR_Stats + 'Frame_Histogram.json', 'r') as file:
        histogram = json.load(file)
    pyplot.hist(list(histogram.values()))
    pyplot.show()


def retrieve_max():
    if(os.path.exists(Frame_DIR_Stats + 'max_frame_size.json') is False):
        print("PLEASE UNCOMMENT THE STATISTICS BLOCK BELOW BEFORE RUNNING ANY OF THIS")
    stats = {}
    with open(Frame_DIR_Stats + 'max_frame_size.json', 'r') as file:
        stats = json.load(file)
    return stats['max']


# ONLY UNCOMMENT IF YOU WANT TO RECOMPUTE THE FRAME STATISTIC INFORMATION
###########################################################################
# raw_training_data = load_raw_data_sliding_window(ADL_Directories)       #
# raw_testing_dta = load_raw_data_sliding_window(Falling_Directories)     #
# training_map = generate_frame_sequences(raw_training_data)              #
# testing_map = generate_frame_sequences(raw_testing_dta)                 #
# compute_frame_statistics(training_map, testing_map)                     #
# print_histogram()                                                       #
###########################################################################
MAX_POINT_FRAME = retrieve_max()
train = load_mmData(ADL_Directories, "Training_Set", Array_DIR_Train)
test = load_mmData(Falling_Directories, "Testing_Set", Array_DIR_Test)
print(np.shape(train))
print(np.shape(test))
#model = train_HVRNNAE(train, test)
# print(model.summary())
