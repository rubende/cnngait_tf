import keras
import Model
import createDatasetRecord
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau
import os
import callbacks
from keras import optimizers



INPUT_PATH = "/inputs_N150/"                    # Input data from TUM-GAID dataset. Into /inputs_N150/ we have 3 folders, one for video type. Into this folders, we stored one tfrecord file per video.
OUTPUT_PATH = "/outputs/"                       # Output path, where we store the resulting model.
    

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

# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.7
###################################


graph = tf.Graph()
with graph.as_default():
    sess = tf.Session(config=config, graph=graph)
    with sess.as_default():

        # First training (150)
        filenames0, filenames1, filenames2, val_filenames0, val_filenames1, val_filenames2 = \
            createDatasetRecord.create_dataset(INPUT_PATH, 0.1)


        len_train = len(filenames0) + len(filenames1) + len(filenames2)
        len_val = len(val_filenames0) + len(val_filenames1) + len(val_filenames2)
        steps_per_epoch = int(len_train / batch)
        validation_steps = int(len_val / batch)

        images, labels = createDatasetRecord.create_tfrecord(batch, filenames0, filenames1, filenames2, 0)

        # Build model
        model = Model.Network.build(input_shape, num_class, number_convolutional_layers,
                                    filters_size, filters_numbers, weight_decay, dropout, lr, momentum)
        w = model.layers[1].get_weights()
        print(w[0][0][0][0][0])

        # Learning Rate update
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                      patience=3, min_lr=0.0001)

        # Create a TensorBoard instance with the path to the logs directory
        tensorboard = callbacks.TrainValTensorBoard_150(histogram_freq=0, write_graph=True, write_images=True)

        # Load dataset
        filenames0, filenames1, filenames2, val_filenames0, val_filenames1, val_filenames2 = \
            createDatasetRecord.create_dataset(INPUT_PATH, 0.1)

        # Create TFRecord to val
        val_images, val_labels = \
            createDatasetRecord.create_tfrecord(batch, val_filenames0, val_filenames1, val_filenames2)

        callbacks = [reduce_lr, tensorboard]

        steps = 0
        epoch = 0
        
        # Incremental Learning
        for i in range(len(filenames0)):
            print("Incremental learning with index " + str(i))
            epoch = epoch + epochs[i]
            # Create TFRecord to train
            images, labels = createDatasetRecord.create_tfrecord(batch, filenames0, filenames1, filenames2, i)

            # Train model
            model = Model.Network.fit(model, epoch, callbacks, images, labels, val_images, val_labels,
                                  steps, steps_per_epoch, validation_steps)

            steps = steps + epochs[i]
            print("------------------------------------------------------------------")

        #Validation data train

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
        model = model.layers[-1].layers[-1].layers[-1].layers[-1].layers[-1]
        model = keras.models.Model(model.layers[0].input, model.layers[-1].output)
        model_input = keras.layers.Input(shape=input_shape)
        newOutputs = model(model_input)
        model = keras.models.Model(model_input, newOutputs)
        model.layers[-1].compile(loss='categorical_crossentropy',
                                 optimizer=optimizers.SGD(lr, 0.9), target_tensors=[val_labels],
                                 metrics=['accuracy'])

        model.layers[-1].save(OUTPUT_PATH+'model_150.h5')


quit()