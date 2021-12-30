import pandas as pd
import numpy as np
import tensorflow as tf
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from datetime import datetime
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.layers import Input, TimeDistributed, Flatten, Lambda, Concatenate, Reshape, LSTM, RepeatVector, SimpleRNN, Activation, Dense
from tensorflow.keras import backend as kb
from tensorflow.keras.models import Model, load_model
INPUT_DIR = "Data_Preprocessed/Radar_Dataset"

epochs = 5

# step 1 collect data into appropriate tensors
# part 1 of data is just the 4d vector [x,y,z,Doppler] <- all supplied in target_list.csv
# can do a basic train/test split of 85%


def load_mmData():
    training_set = []
    testing_set = []
    radar_dataset = os.listdir(INPUT_DIR)
    random.shuffle(radar_dataset)
    test_max = len(radar_dataset) - (int)(0.85*len(radar_dataset))
    for participant_data in radar_dataset:
        cwd = os.getcwd()
        participant_file_path = cwd + "/Data_Preprocessed/Radar_Dataset/" + \
            participant_data + "/points_cloud_clean.csv"
        data = pd.read_csv(participant_file_path)
        # take the values from data needed like stated in paper
        values = data[['x', 'y', 'z', 'doppler']]
        if(test_max > 0):
            test_max -= 1
            testing_set.extend(values.to_numpy())
        else:
            training_set.extend(values.to_numpy())

    return training_set, testing_set


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
        mean_reshape = kb.reshape(predicted_mean, (batch_size, number_of_frames, -1))
        variance_reshape = kb.reshape(predicted_variance, (batch_size, number_of_frames, -1))
        log_variance_reshape = kb.reshape(predicted_log_variance, (batch_size, number_of_frames, -1))

        log_output = (kb.square(true_reshape - mean_reshape))/variance_reshape
        log_output = kb.sum(0.5*log_output, axis=-1)

        KL_loss = -0.5*kb.sum(1 + input_log_variance - kb.square(input_mean) - kb.exp(input_log_variance), axis = -1)
        return kb.mean(log_output + KL_loss)

    model = Model(inputs, outputs)
    optimiser = optimizers.Adam()
    model.compile(optimizer=optimiser, loss=HVRNNAE_loss)
    print(model.summary())
    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = callbacks.TensorBoard(log_dir=logdir)
    model.fit(training_set, training_set, epochs=5, batch_size=8, shuffle=False, validation_data=(testing_set, testing_set), verbose=0, callbacks=[tensorboard_callback])
    print("model finished training")
    return model


def predict(model, inference_data):
    print("#################Predicting####################")
    kb.clear_session()
    output_mean = Model(inputs = model.input, outputs=model.get_layer('input_mean').output)
    output_log_variance = Model(inputs=model.input, outputs=model.get_layer('input_log_variance').output)
    predictions = []
    losses = []
    for data in inference_data:
        data = np.expand_dims(data, axis=0)
        current_prediction = model.predict(data, batch_size=1)
        predicted_output_mean = output_mean.predict(data, batch_size=1)
        predicted_log_variance = output_log_variance.predict(data, batch_size=1)
        current_loss = loss(data, current_prediction, predicted_output_mean, predicted_log_variance)
        losses.append(current_loss)
    return losses
    # model.add()
# step 2 implement the HVRNNAE model described in paper
# def train_HVRNNAE():

def loss(y_t, y_p, output_mean, output_log_variance):
    batch_size = kb.shape(y_t)[0]
    number_of_frames = kb.shape(y_t)[1]
    number_of_features = kb.shape(y_t)[-1]

    predicted_mean = y_p[:, :, :, :number_of_features]
    predicted_log_variance = y_p[:, :, :, number_of_features:]
    predicted_variance = kb.exp(predicted_log_variance)

    true_reshape = kb.reshape(y_t, (batch_size, number_of_frames, -1))
    mean_reshape = kb.reshape(predicted_mean, (batch_size, number_of_frames, -1))
    variance_reshape = kb.reshape(predicted_variance, (batch_size, number_of_frames, -1))
    log_variance_reshape = kb.reshape(predicted_log_variance, (batch_size, number_of_frames, -1))

    log_output = (kb.square(true_reshape - mean_reshape))/variance_reshape
    log_output = kb.sum(0.5*log_output, axis=-1)

    KL_loss = -0.5*kb.sum(1 + output_log_variance - kb.square(output_mean) - kb.exp(output_log_variance), axis = -1)
    return kb.mean(log_output + KL_loss)

def sample(inputs):
    input_mean, input_log_variance = inputs
    batch_size = kb.shape(input_mean)[0]
    number_of_frames = kb.int_shape(input_mean)[1]
    latent_dimension = kb.int_shape(input_mean)[2]
    epsilon = kb.random_normal(shape=(
        batch_size, number_of_frames, latent_dimension), mean=0., stddev=1.0, seed=None)
    return input_mean + kb.exp(0.5*input_log_variance) * epsilon

train, test = load_mmData()
model = train_HVRNNAE(train, test)

#output = predict(model, test)
#print(output)
# step 3 if detected a spike in loss function -> pass through the check to see if it is a fall by using
# the change in the centroid from the spike.
