import os
import sys
import numpy as np
import tensorflow as tf
import glob
from model import dataio, model
from tensorflow import keras

import tensorflow as tf
print(tf.version.VERSION)

# import tensorflow libraries
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, ZeroPadding2D
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, Add, Dense, BatchNormalization, ReLU, MaxPool2D, GlobalAvgPool2D, Conv2D,  GlobalAveragePooling2D, Reshape,Lambda, LSTM, concatenate
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization
from tensorflow.keras.layers import UpSampling2D, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Recall, Precision, categorical_accuracy
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import backend as K
from util import custom_metrics
from tensorflow.keras.applications import MobileNetV3Small, EfficientNetV2S,EfficientNetV2B0, EfficientNetV2B3

def parse_tfrecord(example_proto):
    """The parsing function.
    Read a serialized example into the structure defined by FEATURES_DICT.
    Args:
        example_proto: a serialized Example.
    Returns:
        A dictionary of tensors, keyed by feature name.
    """
    return tf.io.parse_single_example(example_proto, FEATURES_DICT)

def to_tuple(inputs):
    """Function to convert a dictionary of tensors to a tuple of (inputs, outputs).
    Turn the tensors returned by parse_tfrecord into a stack in HWC shape.
    Args:
        inputs: A dictionary of tensors, keyed by feature name.
    Returns:
        A tuple of (inputs, outputs).
    """
    inputsList = [inputs.get(key) for key in FEATURES]
    stacked = tf.stack(inputsList, axis=0)
    # Convert from CHW to HWC
    stacked = tf.transpose(stacked, [1, 2, 0])
    return stacked[:,:,:len(BANDS)], stacked[:,:,len(BANDS):]

def get_dataset(pattern):
    """Function to read, parse and format to tuple a set of input tfrecord files.
    Get all the files matching the pattern, parse and convert to tuple.
    Args:
        pattern: A file pattern to match in a Cloud Storage bucket.
    Returns:
        A tf.data.Dataset
    """
    glob = tf.io.gfile.glob(pattern)
    dataset = tf.data.TFRecordDataset(glob, compression_type='GZIP')
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=8)
    dataset = dataset.map(to_tuple, num_parallel_calls=8)
    return dataset

def get_training_dataset(input_path):
    """Loads the training dataset exported by GEE
    Returns:
        A tf.data.Dataset of training data.
    """
    dataset = get_dataset(input_path+'/training/*')
    return dataset

def get_testing_dataset(input_path):
    """Loads the test dataset exported by GEE
        Returns:
        A tf.data.Dataset of evaluation data.
    """
    dataset = get_dataset(input_path+'/testing/*')
    return dataset



def get_modeling_data_stats(train, test):
    '''
    Computes the channel means and the std of the train and test sets.
    Additionall, the length of the training and testing sets are computed

    Paramters:
        train: a TFRecordDataset for the training set
        test: a TFRecordDataset for the testing set

    Returns:
        mean - a tf.Tensor(4,) containing the channel means
        std - a tf.Tensor(4,) containin the channel standard deviation
        train_len - the number of elements in the training dataset
        test_len - the number of elements in the testing dataset

    '''
    # Initialize the variables that we need to keep track of
    mean = tf.constant([0.,0.,0.,0.])
    std = tf.constant([0.,0.,0.,0.])
    nb_samples = 0.0
    train_len = 0.0
    test_len = 0.0

    # Loop through the training dataset
    for element in train:
        mean = tf.math.add(mean, tf.math.reduce_mean(element[0], axis=[0,1]))
        std = tf.math.add(std, tf.math.reduce_std(element[0], axis=[0,1]))
        nb_samples += 1
        train_len += 1

    # Loop through the testing dataset
    for element in test:
        mean = tf.math.add(mean, tf.math.reduce_mean(element[0], axis=[0,1]))
        std = tf.math.add(std, tf.math.reduce_std(element[0], axis=[0,1]))
        nb_samples += 1
        test_len += 1

    # Divide by the number of elements in the two sets
    mean = tf.math.divide(mean, nb_samples)
    std = tf.math.divide(std, nb_samples)

    return mean, std, train_len, test_len


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    true_sum = K.sum(K.square(y_true), -1)
    pred_sum = K.sum(K.square(y_pred), -1)
    return 1 - ((2. * intersection + smooth) / (true_sum + pred_sum + smooth))

def tversky(y_true, y_pred, smooth=1, alpha=0.7):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    tv = tversky(y_true, y_pred)
    return K.pow((1 - tv), gamma)


def bce_dice_loss(y_true, y_pred):
    return keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def bce_loss(y_true, y_pred):
    return keras.losses.binary_crossentropy(y_true, y_pred, label_smoothing=0.2)

def tversky_bce(y_true, y_pred):
	return focal_tversky_loss(y_true, y_pred) + dice_loss(y_true, y_pred) # keras.losses.binary_crossentropy(y_true, y_pred)


def build_efficientNet():

    inputs = Input(shape=(None,None, 6), name="input_image")
    encoder = EfficientNetV2B0(input_tensor=inputs, weights=None, include_top=False,include_preprocessing =False,classifier_activation=None)

    inp = encoder.input

    skip_connection_names = ["input_image", "block1a_project_activation","block2b_expand_activation","block4a_expand_activation", "block6a_expand_activation"]
    encoder_output = encoder.get_layer("top_activation").output

    f = [16,32, 64, 128, 256]

    x = encoder_output
    for i in range(1, len(skip_connection_names)+1, 1):
        x_skip = encoder.get_layer(skip_connection_names[-i]).output
        x = UpSampling2D((2, 2))(x)
        x = Concatenate()([x, x_skip])

        x = Conv2D(f[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(f[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)


    output = Conv2D(2, (3,3), padding="same", activation="sigmoid")(x)

    model = Model(inputs=[inp], outputs=[output], name="unet")
    return model


def get_models(input_optimizer, input_loss_function, evaluation_metrics):
    model = build_efficientNet()

    model.summary()

    model.compile(
        optimizer = input_optimizer,
        loss = input_loss_function,
        metrics = evaluation_metrics
        )


    return model


if __name__ == "__main__":

    # Set the path to the raw data
    raw_data_path = r"/home/ate/sig/alerts/models/data/alertsMKDescV5"

    # Define the path to the log directory for tensorboard
    log_dir = r'/home/ate/sig/alerts/models/log'

    # Define the directory where the models will be saved
    model_dir = r'/home/ate/sig/palawan/efficientModelSAR'

    # for sentinel 1 data
    BANDS =  ['VH_after0', 'VH_before0', 'VH_before1', 'VV_after0','VV_before0', 'VV_before1']

    # for NICFI data
    #BANDS =  ["rb","gb","bb","nb","ra","ga","ba","na"]

    RESPONSE = ["alert","other"]
    FEATURES = BANDS + RESPONSE

    # Specify model training parameters.
    #TRAIN_SIZE = 550000
    TRAIN_SIZE = 160000
    BATCH_SIZE = 32
    EPOCHS = 70
    BUFFER_SIZE = 1024*4
    optimizer = "Adam"

    eval_metrics = [categorical_accuracy, custom_metrics.f1_m, custom_metrics.precision_m, custom_metrics.recall_m]

    # Specify the size and shape of patches expected by the model.
    kernel_size = 128
    kernel_shape = [kernel_size, kernel_size]
    COLUMNS = [tf.io.FixedLenFeature(shape=kernel_shape, dtype=tf.float32) for k in FEATURES]
    FEATURES_DICT = dict(zip(FEATURES, COLUMNS))

    KERNEL_SIZE = 128
    PATCH_SHAPE = (KERNEL_SIZE, KERNEL_SIZE)

    training_files = glob.glob(raw_data_path + '/training/*')
    training_ds = dataio.get_dataset(training_files, BANDS, RESPONSE, PATCH_SHAPE, BATCH_SIZE, buffer_size=BUFFER_SIZE, training=True).repeat()

    testing_files = glob.glob(raw_data_path + '/testing/*')
    testing_ds = dataio.get_dataset(training_files, BANDS, RESPONSE, PATCH_SHAPE, BATCH_SIZE,buffer_size=BUFFER_SIZE)


    val_files = glob.glob(raw_data_path + '/validation/*')
    val_ds = dataio.get_dataset(training_files, BANDS, RESPONSE, PATCH_SHAPE, BATCH_SIZE,buffer_size=BUFFER_SIZE)

    model = get_models(optimizer,tversky_bce, eval_metrics)

    #model.load_weights(r'/home/ate/sig/alerts/models/weights/modelkhEfficientv1desc.h5',by_name=True,skip_mismatch=True)

    #tensorboard = callbacks.TensorBoard(log_dir=log_dir)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, verbose=0, mode='min',restore_best_weights=True)

    model.fit(
        x = training_ds,
        epochs = EPOCHS,
        steps_per_epoch =int(TRAIN_SIZE / BATCH_SIZE),
        validation_data = val_ds,
        validation_steps = 500,
        callbacks=[early_stop]
        )
    # check how the model trained
    print(model.evaluate(val_ds))

    # Save the model
    model.save(model_dir, save_format='tf')
    model.save_weights( r'/home/ate/sig/palawan/allWeightsSARv1', save_format='tf')
    model.save_weights( r'/home/ate/sig/palawan/allWeightsSARv1.h5')
