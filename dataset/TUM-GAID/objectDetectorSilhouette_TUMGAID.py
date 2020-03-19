## Based in code from https://averdones.github.io/real-time-semantic-image-segmentation-with-deeplab-in-tensorflow/

## Imports

import collections
import os
import sys
import tarfile
import tempfile
import urllib
import numpy as np
from PIL import Image
import glob
import cv2
import ntpath
import tensorflow as tf

PATH_TO_RESEARCH = '/tensorflow/models/research/'                   # Path to /tensorflow/models/research/ installation folder

PATH_TO_TEST_IMAGES_DIR = '/TUM_GAID/image'                         # Path to the original images of the dataset

OUTPUT_PATH = '/TUM_GAID/silhouettes/'                              # Output path


# if tf.__version__ < '1.5.0':
#    raise ImportError('Please upgrade your tensorflow installation to v1.5.0 or newer!')

# Needed to show segmentation colormap labels
sys.path.append(PATH_TO_RESEARCH)
from deeplab.utils import get_dataset_colormap

## Select and download models

_MODEL_URLS = {
    'xception_coco_voctrainaug': 'http://download.tensorflow.org/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
    'xception_coco_voctrainval': 'http://download.tensorflow.org/models/deeplabv3_pascal_trainval_2018_01_04.tar.gz',
}

# Download the model

_TARBALL_NAME = 'deeplab_model.tar.gz'
# model_dir = config.model_dir or tempfile.mkdtemp()

# config = 'http://download.tensorflow.org/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz' # nueva
model_dir = tempfile.mkdtemp()
model_url = 'http://download.tensorflow.org/models/deeplabv3_pascal_trainval_2018_01_04.tar.gz'
tf.gfile.MakeDirs(model_dir)

download_path = os.path.join(model_dir, _TARBALL_NAME)
print('downloading model to %s, this might take a while...' % download_path)
urllib.request.urlretrieve(model_url, download_path)
print('download completed!')

## Load model in TensorFlow

_FROZEN_GRAPH_NAME = 'frozen_inference_graph'


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if _FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break

        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.

        Args:
            image: A PIL.Image object, raw input image.

        Returns:
            resized_image: RGB image resized from original input image.
            seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map


model = DeepLabModel(download_path)

## Helper methods

LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
    'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
    'train', 'tv'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = get_dataset_colormap.label_to_color_image(FULL_LABEL_MAP)



TEST_IMAGE_PATHS = [f for f in glob.glob(PATH_TO_TEST_IMAGES_DIR + "/**/*.jpg", recursive=True)]


for image_path in TEST_IMAGE_PATHS:
    image = Image.open(image_path)
    resized_im, seg_map = model.run(image)
    seg_image = get_dataset_colormap.label_to_color_image(
        seg_map, get_dataset_colormap.get_pascal_name()).astype(np.uint8)
    
    basename = image_path.replace(PATH_TO_TEST_IMAGES_DIR,"")
    open_cv_image = np.array(seg_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    os.makedirs(OUTPUT_PATH + basename.replace(ntpath.basename(image_path), ""),
                exist_ok=True)
    cv2.imwrite(OUTPUT_PATH + basename, open_cv_image)
