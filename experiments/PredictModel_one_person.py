from keras.models import load_model
import tensorflow as tf
import os
import numpy as np
import glob
import createDatasetRecord
import keras
from keras import optimizers
from scipy import stats
import math



PATH_ID_FILE = "tumgaidtestids.lst"                             # File with the IDs of the users that we are going to use
PATH_25F_INPUT = '/MulPerGait_one_person_cnn_25f/'              # Path with samples windows with 25 frame
PATH_MODEL_CNN = "/outputs/model_155.h5"                        # Model_155 path




###################################
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # To force tensorflow to only see one GPU.
# TensorFlow wizardry
config = tf.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True

# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.7
###################################


filenames0 = [f for f in glob.glob(PATH_25F_INPUT + "/*.record", recursive=True)]
unicos = np.loadtxt(PATH_ID_FILE).astype(int)


batch = 128

ids = []
for f in filenames0:
    ids.append(f[:-16])
ids = np.unique(ids)

graph = tf.Graph()
with graph.as_default():
    sess = tf.Session(config=config, graph=graph)
    # with tf.device('/gpu:' + str(1)):
    with sess.as_default():
        images = createDatasetRecord.create_tfrecord_155_predict(batch, filenames0)
        model = load_model(PATH_MODEL_CNN)
        lr = model.optimizer.lr
        model.layers.pop(0)
        newInput = keras.layers.Input(tensor=images)
        newOutput = model(newInput)
        model = keras.models.Model(inputs=newInput, outputs=newOutput)
        model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr, 0.9),
                      metrics=['accuracy'])
        model.summary()

        t = model.predict(images, steps=math.ceil(len(filenames0)/batch))


results = []
_T0 = []
_T1 = []
file_names = []

contador_moda = 0
for id in ids:
    files = [s for s in filenames0 if id in s]

    label1 = np.where(unicos == int(id[-6:-3]))

    if len(files) == 0:
        results.append(0)
        continue

    all_index_0 = []
    for f in files:
        index_0 = [i for i, s in enumerate(filenames0) if f in s]
        all_index_0.append(index_0[0])

    T = [t[i] for i in all_index_0]
    T = np.argmax(T, 1)
    T = stats.mode(T)[0][0]
    T0 = unicos[T]
    _T0.append(T0)


    file_names.append(files[0][-23:-9])

    if (T0 == unicos[label1][0]):
        results.append(1)
    else:
        results.append(0)

print(sum(results)/len(results))
quit()