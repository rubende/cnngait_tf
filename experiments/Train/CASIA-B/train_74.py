import keras
import tensorflow as tf
import os
import random
from numpy.random import seed
from tensorflow import set_random_seed
import math
import importlib

ModelCasiaB = importlib.import_module('cnngait_tf.experiments.Common.Model')
createDatasetRecordCasiaB = importlib.import_module('cnngait_tf.experiments.Common.createDatasetRecord')
callbacksCasiaB = importlib.import_module('cnngait_tf.experiments.Common.callbacks')



INPUT_PATH = "inputs_N74/"                     # Input data from CASIA-B dataset. Into /inputs_N74/ we have 3 folders, one for video type. Into this folders, we stored one tfrecord file per video.
OUTPUT_PATH = "/outputs/"                       # Output path, where we store the resulting model.
LOG_DIR = "/logs/casia_74/"                     # Tensorboard log path

# Hyperparameters
mean_dataset = 0.2202
input_shape = (50, 60, 60)
num_class = 74
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


def step_decay(epoch):
    if epoch >= 8:
        return float(lr/10)
    else:
        return float(lr)


graph = tf.Graph()
with graph.as_default():
    sess = tf.Session(config=config, graph=graph)
    with sess.as_default():

        # First training (74)
        filenames0, filenames1, filenames2, val_filenames0, val_filenames1, val_filenames2 = \
            createDatasetRecordCasiaB.create_dataset(INPUT_PATH, 0.1)

        steps_per_epoch = math.ceil((len(filenames0[0])*3) / batch)
        validation_steps = math.ceil((len(val_filenames0[0])*3) / batch)

        # Build model
        model = ModelCasiaB()
        model.build(input_shape, num_class, number_convolutional_layers, filters_size,
                                    filters_numbers, weight_decay, dropout)

        # Learning Rate update
        change_lr = keras.callbacks.LearningRateScheduler(step_decay)

        # Create a TensorBoard instance with the path to the logs directory
        tensorboard = callbacksCasiaB.TrainValTensorBoard(log_dir=LOG_DIR, histogram_freq=0, write_graph=True, write_images=True)

        # Create TFRecord to val
        val_images, val_labels = \
            createDatasetRecordCasiaB.create_tfrecord(batch, num_class, val_filenames0, val_filenames1, val_filenames2, mean=mean_dataset)

        callbacks = [tensorboard, change_lr]

        steps = 0
        epoch = 0
        # Incremental Learning
        for i in range(len(filenames0)):
            print("Incremental learning with index " + str(i))
            epoch = epoch + epochs[i]
            # Create TFRecord to train
            images, labels = createDatasetRecordCasiaB.create_tfrecord(batch, num_class, filenames0, filenames1, filenames2, i, mean=mean_dataset)

            # Train model
            model.fit(epoch, callbacks, images, labels, val_images, val_labels,
                                  steps, steps_per_epoch, validation_steps, lr, momentum)

            steps = steps + epochs[i]
            print("------------------------------------------------------------------")

        #Validation data train
        model.fit_val(epoch, callbacks, val_images, val_labels, steps, steps_per_epoch, lr, momentum)

        model.save(OUTPUT_PATH+'model_casia_74.h5')


quit()