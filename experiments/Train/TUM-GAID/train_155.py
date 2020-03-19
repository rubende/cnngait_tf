import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau
import os
import random
from numpy.random import seed
from tensorflow import set_random_seed
import math
import importlib

ModelTUM = importlib.import_module('cnngait_tf.experiments.Common.Model')
createDatasetRecordTUM = importlib.import_module('cnngait_tf.experiments.Common.createDatasetRecord')
callbacksTUM = importlib.import_module('cnngait_tf.experiments.Common.callbacks')


model_150_path = "/outputs/model_150.h5"        # Model_150 path
INPUT_PATH = "/inputs_N155/"                    # Input data from TUM-GAID dataset. Into /inputs_N155/ we have 3 folders, one for video type. Into this folders, we stored one tfrecord file per video.
OUTPUT_PATH = "/outputs/"                       # Output path, where we store the resulting model.
LOG_DIR = "./logs/TUM_155/"                    # Tensorboard log path


# Hyperparameters
input_shape = (50, 60, 60)
num_class = 150
number_convolutional_layers = 4
filters_size = [(7,7), (5,5), (3,3), (2,2)]
filters_numbers = [96, 192, 512, 512]
weight_decay=0.00005
dropout=0.40
lr=0.001
momentum=0.9
epochs = [15, 5, 5, 5]
batch = 128
validation = 0.1



###################################
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # To force tensorflow to only see one GPU.
# TensorFlow wizardry
config = tf.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
###################################

seed(0)
set_random_seed(0)
random.seed(0)

graph = tf.Graph()
with graph.as_default():
    sess = tf.Session(config=config, graph=graph)
    with sess.as_default():
        # Learning Rate update
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                      patience=3, min_lr=0.0001)

        # Create a TensorBoard instance with the path to the logs directory
        tensorboard = callbacksTUM.TrainValTensorBoard(log_dir=LOG_DIR, histogram_freq=0, write_graph=True,
                                                          write_images=True)

        callbacks = [reduce_lr, tensorboard]


        # Training 155
        print("Second training")
        filenames0, val_filenames0 = \
            createDatasetRecordTUM.create_dataset_155(INPUT_PATH, 0.1)

        # Create TFRecord to val
        val_images, val_labels = \
            createDatasetRecordTUM.create_tfrecord_ft(batch, num_class, val_filenames0)

        # Create TFRecord to train
        images, labels = createDatasetRecordTUM.create_tfrecord_ft(batch, num_class, filenames0)

        steps_per_epoch = math.ceil((len(filenames0)) / batch)
        validation_steps = math.ceil(len(val_filenames0) / batch)

        # Load model to FT, freeze layers and modify FC classification layer
        model = ModelTUM.Network()
        model.load_to_ft(model_150_path, num_class, dropout)

        # Train model
        epoch = sum(epochs)
        epoch = epoch + epochs[0]
        steps = sum(epochs)
        model.fit(epoch, callbacks, images, labels, val_images, val_labels,
                  steps, steps_per_epoch, validation_steps, lr, momentum)
        steps = steps + epochs[0]

        # Validation data train
        epoch = epoch + epochs[-1]
        model.fit_val(epoch, callbacks, val_images, val_labels, steps, steps_per_epoch, lr, momentum)

        model.save(OUTPUT_PATH + 'model_casia_50.h5')
