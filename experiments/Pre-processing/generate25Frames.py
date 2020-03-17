import cv2
import glob
import numpy as np
import scipy.io
import ntpath
import pickle
import tensorflow as tf


PATH_TO_OF = '/video_two_persons_of/'              # Path to optical flow of the generated dataset

PATH_TO_TR = '/video_two_persons_cnn_tr/'          # Path to tracking information of the generated dataset

PATH_TO_IMAGE = '/video_two_persons/'              # Path to generated dataset

OUTPUT_PATH = '/video_two_persons_cnn_25f/'        # Output path

n_frames = 25       # Number of frames to stack

# Size of the images.
im_height = 480
im_width = 640
    

# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_tf_example(data):

  height = 60                   # Image height
  width = 60                    # Image width
  depth = 50
  compressFactor = 100

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'height': _int64_feature(height),
      'width': _int64_feature(width),
      'depth': _int64_feature(depth),
      'data': _bytes_feature(data),
      'compressFactor': _int64_feature(compressFactor),
  }))
  return tf_example


np.random.seed(0)

x_scale = 80 / im_width
y_scale = 60 / im_height

paths_of = sorted([f.replace(PATH_TO_OF, "") for f in glob.glob(
            PATH_TO_OF + "/*.mat",
            recursive=True)])

paths_tr = sorted([f.replace(PATH_TO_TR, "") for f in glob.glob(
            PATH_TO_TR + "/*.pkl",
            recursive=True)])

for file in paths_of:
    of_file = scipy.io.loadmat(PATH_TO_OF+file)
    basename = ntpath.basename(file)


    with open(PATH_TO_TR + file[:-10] + '.pkl', 'rb') as f:
        [detection_list, position_list] = pickle.load(f)

    of_output = []
    for i in range(0, len(of_file['S']['frix'][0]), 5):
        frames = of_file['S']['frix'][0][i:i+n_frames][0][0]
        ofs = of_file['S']['of'][0][i:i+n_frames]
        sub_detection_list = detection_list[i:i+n_frames]
        sub_position_list = position_list[i:i+n_frames]

        centroides_bb = []
        bb = []
        of_resizes = []

        if len(ofs) < n_frames:
            continue

        for j in range(len(ofs)):
            of_resizes.append(cv2.resize(ofs[j], (80, 60)))       


            bb_temp = np.zeros([2, 4])
            centroides_bb_temp = np.zeros([2, 2])

            for k in range(len(sub_detection_list[j])):                                 
                if (sub_detection_list[j][k] != 0) and sub_detection_list[j][k] < 3:
                    x = int(np.round(sub_position_list[j][k][1] * x_scale))            
                    y = int(np.round(sub_position_list[j][k][0] * y_scale))
                    xmax = int(np.round(sub_position_list[j][k][3] * x_scale))
                    ymax = int(np.round(sub_position_list[j][k][2] * y_scale))

                    print(basename + "_" + str(i) + "_" + str(j) + "_" + str(k))
                    bb_temp[int(sub_detection_list[j][k]) - 1] = [x, y, xmax, ymax]
                    centroides_bb_temp[int(sub_detection_list[j][k]) - 1] = [(y+ymax)/2, (x+xmax)/2]

            bb.append(bb_temp)
            centroides_bb.append(centroides_bb_temp)

        of_resizes_0 = []
        of_resizes_1 = []


        exist = True
        for j in range(len(sub_detection_list)):
            if 1 not in sub_detection_list[j]:
                exist = False

        if exist:
            dif_bb = 30 - centroides_bb[round(n_frames/2)][0][1]
            M = np.float32([[1, 0, dif_bb], [0, 1, 0]])
            of_resizes_0 =  np.zeros([60, 60, 50])
            for k in range(len(of_resizes)):
                imagetrans = np.zeros([60, 80, 2])
                imagetrans[int(bb_1[1]):int(bb_1[3]), int(bb_1[0]):int(bb_1[2]), :] = of_resizes[k][int(bb_1[1]):int(bb_1[3]), int(bb_1[0]):int(bb_1[2]), :]
                imagetrans= cv2.warpAffine(imagetrans, M, (60, 60))
                of_resizes_0[:,:,2*k:2*k+2] = imagetrans

        exist = True
        for j in range(len(sub_detection_list)):
            if 2 not in sub_detection_list[j]:
                exist = False

        if exist:
            dif_bb = 30 - centroides_bb[round(n_frames/2)][1][1]
            M = np.float32([[1, 0, dif_bb], [0, 1, 0]])
            of_resizes_1 = np.zeros([60, 60, 50])
            for k in range(len(of_resizes)):
                bb_1 = bb[k][1]
                imagetrans = np.zeros([60, 80, 2])
                imagetrans[int(bb_1[1]):int(bb_1[3]), int(bb_1[0]):int(bb_1[2]), :] = of_resizes[k][int(bb_1[1]):int(bb_1[3]),int(bb_1[0]):int(bb_1[2]), :]
                imagetrans= cv2.warpAffine(of_resizes[k], M, (60, 60))
                of_resizes_1[:, :, 2 * k:2 * k + 2] = imagetrans



        if len(of_resizes_0) != 0:
            of_resizes_0 = np.swapaxes(of_resizes_0, 0, 2)
            of_resizes_0 = np.swapaxes(of_resizes_0, 1, 2)
            writer = tf.python_io.TFRecordWriter(
                OUTPUT_PATH + file[:-10] + '_' + str(i)
                .zfill(6) + '_0.record')
            of_resizes_0 = of_resizes_0.astype(np.int16)
            tf_example = create_tf_example(tf.compat.as_bytes(of_resizes_0.tostring()))
            writer.write(tf_example.SerializeToString())
            writer.close()

        if len(of_resizes_1) != 0:
            of_resizes_1 = np.swapaxes(of_resizes_1, 0, 2)
            of_resizes_1 = np.swapaxes(of_resizes_1, 1, 2)
            writer_2 = tf.python_io.TFRecordWriter(
                OUTPUT_PATH + file[:-10] + '_' + str(i)
                .zfill(6) + '_1.record')
            of_resizes_1 = of_resizes_1.astype(np.int16)
            tf_example_2 = create_tf_example(tf.compat.as_bytes(of_resizes_1.tostring()))
            writer_2.write(tf_example_2.SerializeToString())
            writer_2.close()
