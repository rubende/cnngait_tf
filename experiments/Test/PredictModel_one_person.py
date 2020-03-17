import tensorflow as tf
import os
import numpy as np
import glob
from scipy import stats
import math
import importlib
createDatasetRecord = importlib.import_module('cnngait_tf.experiments.Common.createDatasetRecord')
Model = importlib.import_module('cnngait_tf.experiments.Common.Model')



PATH_ID_FILE = "tumgaidtestids.lst"                             # File with the IDs of the users that we are going to use
PATH_25F_INPUT = '/videos_one_person_cnn_25f/'              # Path with samples windows with 25 frame
PATH_MODEL_CNN = "/outputs/model.h5"                        # Model_155 path




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
        images = createDatasetRecord.create_tfrecord_predict(batch, filenames0)
        model = Model()
        t = model.load_to_predict(PATH_MODEL_CNN, images, math.ceil(len(filenames0)/batch))

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