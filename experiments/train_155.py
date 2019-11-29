import keras
import Model
import createDatasetRecord
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau
import os
import callbacks
from keras.layers import Dense, Dropout
from keras import optimizers
from keras.models import load_model



model_150_path = "/outputs/model_150.h5"        # Model_150 path
INPUT_PATH = "/inputs_N155/"                    # Input data from TUM-GAID dataset. Into /inputs_N155/ we have 3 folders, one for video type. Into this folders, we stored one tfrecord file per video.
OUTPUT_PATH = "/outputs/"                       # Output path, where we store the resulting model.


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

# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.7
###################################


graph = tf.Graph()
with graph.as_default():
    sess = tf.Session(config=config, graph=graph)
    with sess.as_default():
        # Learning Rate update
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                      patience=3, min_lr=0.0001)

        # Create a TensorBoard instance with the path to the logs directory
        tensorboard = callbacks.TrainValTensorBoard_155(histogram_freq=0, write_graph=True, write_images=True)

        callbacks = [reduce_lr, tensorboard]


        # Training 155
        print("Second training")
        filenames0, val_filenames0 = \
            createDatasetRecord.create_dataset_155(INPUT_PATH, 0.1)

        # Create TFRecord to val
        val_images, val_labels = \
            createDatasetRecord.create_tfrecord_155(batch, val_filenames0)

        # Create TFRecord to train
        images, labels = createDatasetRecord.create_tfrecord_155(batch, filenames0)

        steps_per_epoch = int(len(filenames0) / batch)
        validation_steps = int(len(val_filenames0) / batch)

        # Load Model 150
        model = load_model(model_150_path)

        # Freeze the weights
        for layer in model.layers:
            layer.trainable = False

        # Change FC
        lr_temp = model.optimizer.lr

        model.layers.pop()
        model.layers.pop()
        x = Dense(155, activation='softmax', name='id')(model.layers[-1].output)
        x = Dropout(dropout)(x)
        model = keras.models.Model(model.layers[0].input, x)

        model_input = keras.layers.Input(shape=input_shape)
        newOutputs = model(model_input)
        model = keras.models.Model(model_input, newOutputs)

        model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(0.01, 0.9), metrics=['accuracy'])

        model.summary()
        # Train model
        epoch = sum(epochs)
        epoch = epoch + epochs[0]
        steps = sum(epochs)
        model = Model.Network.fit(model, epoch, callbacks, images, labels, val_images, val_labels,
                          steps, steps_per_epoch, validation_steps)
        steps = steps + epochs[0]

        # Validation data train
        lr = model.optimizer.lr

        model.layers.pop(0)
        newInput = keras.layers.Input(tensor=val_images)
        newOutput = model(newInput)
        model = keras.models.Model(inputs=newInput, outputs=newOutput)
        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr, 0.9), target_tensors=[val_labels],
              metrics=['accuracy'])
        epoch = epoch + epochs[-1]
        model.fit(model, epochs=epoch, steps_per_epoch=steps_per_epoch, callbacks=callbacks,
          initial_epoch=steps)

        steps = steps + epochs[-1]

        lr = model.optimizer.lr
        model.layers[-1].layers[-1].layers[-1].compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr, 0.9),
                                               target_tensors=[val_labels],
                                               metrics=['accuracy'])

        lr = model.optimizer.lr
        model.layers[-1].layers[-1].layers[-1].compile(loss='categorical_crossentropy',
                                                       optimizer=optimizers.SGD(lr, 0.9), target_tensors=[val_labels],
                                                       metrics=['accuracy'])

        model.layers[-1].layers[-1].layers[-1].save(OUTPUT_PATH + 'model_155.h5')
