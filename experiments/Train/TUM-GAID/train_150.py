import keras
import tensorflow as tf
import os
import random
from numpy.random import seed
from tensorflow import set_random_seed
import math
import importlib

ModelTUM = importlib.import_module('cnngait_tf.experiments.Common.Model')
createDatasetRecordTUM = importlib.import_module('cnngait_tf.experiments.Common.createDatasetRecord')
callbacksTUM = importlib.import_module('cnngait_tf.experiments.Common.callbacks')



INPUT_PATH = "/inputs_N150/"                    # Input data from TUM-GAID dataset. Into /inputs_N150/ we have 3 folders, one for video type. Into this folders, we stored one tfrecord file per video.
OUTPUT_PATH = "/outputs/"                       # Output path, where we store the resulting model.
LOG_DIR = "/logs/TUM_150/"                     # Tensorboard log path

# Hyperparameters
input_shape = (50, 60, 60)
num_class = 150
number_convolutional_layers = 4
filters_size = [(7,7), (5,5), (3,3), (2,2)]
filters_numbers = [96, 192, 512, 512]
weight_decay=0.00005
dropout=0.40
lr=0.01
momentum=0.9
epochs = [15, 3, 3, 3]
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

        # First training (150)
        filenames0, filenames1, filenames2, val_filenames0, val_filenames1, val_filenames2 = \
            createDatasetRecordTUM.create_dataset(INPUT_PATH, 0.1)

        steps_per_epoch = math.ceil((len(filenames0[0]) * 3) / batch)
        validation_steps = math.ceil((len(val_filenames0[0]) * 3) / batch)


        # Build model
        model = ModelTUM()
        model.build(input_shape, num_class, number_convolutional_layers, filters_size,
                    filters_numbers, weight_decay, dropout)

        # Learning Rate update
        reduce_lr = keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                      patience=3, min_lr=0.0001)

        # Create a TensorBoard instance with the path to the logs directory
        tensorboard = callbacksTUM.TrainValTensorBoard(log_dir=LOG_DIR, histogram_freq=0, write_graph=True, write_images=True)

        # Create TFRecord to val
        val_images, val_labels = \
            createDatasetRecordTUM.create_tfrecord(batch, val_filenames0, val_filenames1, val_filenames2)

        callbacks = [reduce_lr, tensorboard]

        steps = 0
        epoch = 0
        # Incremental Learning
        for i in range(len(filenames0)):
            print("Incremental learning with index " + str(i))
            epoch = epoch + epochs[i]
            # Create TFRecord to train
            images, labels = createDatasetRecordTUM.create_tfrecord(batch, num_class, filenames0, filenames1, filenames2, i)

            # Train model
            model.fit(epoch, callbacks, images, labels, val_images, val_labels,
                      steps, steps_per_epoch, validation_steps, lr, momentum)

            steps = steps + epochs[i]
            print("------------------------------------------------------------------")

        #Validation data train
        model.fit_val(epoch, callbacks, val_images, val_labels, steps, steps_per_epoch, lr, momentum)

        model.save(OUTPUT_PATH + 'model_TUM_150.h5')


quit()