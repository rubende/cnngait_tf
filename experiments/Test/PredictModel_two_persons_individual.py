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
PATH_25F_INPUT = '/MulPerGait_two_persons_cnn_25f/'             # Path with samples windows with 25 frame
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

batch = 128

filenames0 = [f for f in glob.glob(PATH_25F_INPUT + "/*.record", recursive=True)]
unicos = np.loadtxt(PATH_ID_FILE).astype(int)



filenames_temp = []                                 # Use this to filter by type video, or comment to not filter
for file in filenames0:
    if file[-27] == "s":
        filenames_temp.append(file)
filenames0 = filenames_temp



ids = []
for f in filenames0:
    ids.append(f[:-16])
ids = np.unique(ids)

graph = tf.Graph()
with graph.as_default():
    sess = tf.Session(config=config, graph=graph)
    with sess.as_default():
        images = createDatasetRecord.create_tfrecord_predict(batch, filenames0)
        model = Model()
        t = model.load_to_predict(PATH_MODEL_CNN, images, math.ceil(len(filenames0) / batch))


results = []
_T0 = []
_T1 = []
file_names = []

contador_moda = 0
for id in ids:
    files = [s for s in filenames0 if id in s]

    label1 = np.where(unicos == int(id[-14:-11]))
    label2 = np.where(unicos == int(id[-6:-3]))


    files0 = [s for s in files if int(s[-8]) == 0]
    files1 = [s for s in files if int(s[-8]) == 1]


    if len(files0) == 0 or len(files1) == 0:
        results.append(0)
        continue

    all_index_0 = []
    for f in files0:
        index_0 = [i for i, s in enumerate(filenames0) if f in s]
        all_index_0.append(index_0[0])

    T = [t[i] for i in all_index_0]
    T = np.argmax(T, 1)
    T = stats.mode(T)[0][0]
    T0 = unicos[T]
    _T0.append(T0)

    all_index_1 = []
    for f in files1:
        index_1 = [i for i, s in enumerate(filenames0) if f in s]
        all_index_1.append(index_1[0])

    T = [t[i] for i in all_index_1]
    T = np.argmax(T, 1)
    T = stats.mode(T)[0][0]
    T1 = unicos[T]
    _T1.append(T1)

    file_names.append(files0[0][44:59])

    if (T0 == unicos[label1][0] and T1 == unicos[label2][0]) or (T0 == unicos[label2][0] and T1 == unicos[label1][0]):
        results.append(1)
        results.append(1)
    elif T0 == unicos[label1][0] or T0 == unicos[label2][0] or T1 == unicos[label2][0] or T1 == unicos[label1][0]:
        results.append(1)
        results.append(0)
    else:
        results.append(0)
        results.append(0)

print(sum(results)/len(results))
quit()