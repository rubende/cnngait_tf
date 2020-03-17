import tensorflow as tf
import glob
import random
import math


def parse_fn(example, mean=None):
    "Parse TFExample records and perform simple data augmentation."
    image_feature_description = {
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'depth': tf.FixedLenFeature([], tf.int64),
        'data': tf.FixedLenFeature([], tf.string),
        'labels': tf.FixedLenFeature([], tf.int64),
        'set': tf.FixedLenFeature([], tf.int64),
        'videoId': tf.FixedLenFeature([], tf.int64),
        'compressFactor': tf.FixedLenFeature([], tf.int64),
        'gait': tf.FixedLenFeature([], tf.int64),
        'mirrors': tf.FixedLenFeature([], tf.int64),

    }

    parsed = tf.parse_example(example, image_feature_description)

    parsed['data'] = tf.decode_raw(parsed['data'], tf.int16)
    if mean is not None:
        parsed['data'] = tf.math.subtract(tf.cast(parsed['data'], tf.float32), mean)
    parsed['data'] = tf.math.divide(tf.cast(parsed['data'], tf.float32), 100.0)

    return parsed['data'], parsed["labels"], parsed['videoId']



def create_dataset(path, percentaje_val):
    random.seed(0)
    tf.random.set_random_seed(0)
    filenames0 = [f for f in glob.glob(path + "0/" + "**/*.record", recursive=True)] 
    filenames1 = [f for f in glob.glob(path + "1/" + "**/*.record", recursive=True)]
    filenames2 = [f for f in glob.glob(path + "2/" + "**/*.record", recursive=True)]


    random.shuffle(filenames0)
    random.shuffle(filenames1)
    random.shuffle(filenames2)


    total = len(filenames0) + len(filenames1) + len(filenames2)
    val_size = total * percentaje_val
    val_filenames0 = filenames0[:math.floor(val_size/3)]
    filenames0 = filenames0[math.floor(val_size/3):]
    val_filenames1 = filenames1[:math.floor(val_size/3)]
    filenames1 = filenames1[math.floor(val_size/3):]
    val_filenames2 = filenames2[:math.floor(val_size/3)]
    filenames2 = filenames2[math.floor(val_size/3):]

    n = min(len(filenames0), len(filenames1), len(filenames2))
    filenames0 = [filenames0[i * n:(i + 1) * n] for i in range((len(filenames0) + n - 1) // n)]
    filenames1 = [filenames1[i * n:(i + 1) * n] for i in range((len(filenames1) + n - 1) // n)]
    filenames2 = [filenames2[i * n:(i + 1) * n] for i in range((len(filenames2) + n - 1) // n)]


    if len(filenames0[-1]) < n:
        temp = n - len(filenames0[-1])
        filenames0[-1] = filenames0[-1] + filenames0[-2][:temp]

    if len(filenames1[-1]) < n:
        temp = n - len(filenames1[-1])
        filenames1[-1] = filenames1[-1] + filenames1[-2][:temp]

    if len(filenames2[-1]) < n:
        temp = n - len(filenames2[-1])
        filenames2[-1] = filenames2[-1] + filenames2[-2][:temp]


    return filenames0, filenames1, filenames2, val_filenames0, val_filenames1, val_filenames2




def create_dataset_ft(path, percentaje_val):
    random.seed(0)
    tf.random.set_random_seed(0)
    filenames0 = [f for f in glob.glob(path + "0/" + "**/*.record", recursive=True)]

    random.shuffle(filenames0)


    total = len(filenames0)
    val_size = total * percentaje_val
    val_filenames0 = filenames0[:math.floor(val_size/3)]
    filenames0 = filenames0[math.floor(val_size/3):]

    return filenames0, val_filenames0




def create_tfrecord(batch, num_class, filenames0, filenames1, filenames2, index = None, mean = None):

    if index != None:
        index_0 = index % len(filenames0)
        index_1 = index % len(filenames1)
        index_2 = index % len(filenames2)

        files = (filenames0[index_0] + filenames1[index_1] + filenames2[index_2])
    else:
        files = (filenames0 + filenames1 + filenames2)

    size = len(files)

    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.shuffle(buffer_size=(size))
    dataset = dataset.repeat()
    dataset = dataset.prefetch(int(size/batch))
    dataset = dataset.batch(batch_size=batch)
    dataset = dataset.map(lambda x: parse_fn(x, mean), num_parallel_calls=32)
    iter = dataset.make_one_shot_iterator()
    image, label, videoId = iter.get_next()
    #image = tf.strings.to_number(string_tensor=image, out_type=tf.dtypes.float32) / 100
    image = tf.reshape(image, shape=(-1, 50, 60, 60))
    label = tf.one_hot(label, num_class)

    return image, label




def create_tfrecord_ft(batch, num_class, filenames0, mean = None):

    dataset = tf.data.TFRecordDataset(filenames0)
    dataset = dataset.shuffle(buffer_size=(len(filenames0)))
    dataset = dataset.repeat()
    dataset = dataset.prefetch(int(len(filenames0)/batch))
    dataset = dataset.batch(batch_size=batch)
    dataset = dataset.map(lambda x: parse_fn(x, mean), num_parallel_calls=32)
    iter = dataset.make_one_shot_iterator()
    image, label, videoId = iter.get_next()
    image = tf.reshape(image, shape=(-1, 50, 60, 60))
    label = tf.one_hot(label, num_class)

    return image, label


def create_tfrecord_predict(batch, filenames0, mean = None):

    dataset = tf.data.TFRecordDataset(filenames0)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(int(len(filenames0)/batch))
    dataset = dataset.batch(batch_size=batch)
    dataset = dataset.map(lambda x: parse_fn(x, mean), num_parallel_calls=32)
    iter = dataset.make_one_shot_iterator()
    image, label, videoId = iter.get_next()
    image = tf.reshape(image, shape=(-1, 50, 60, 60))

    return image











#######################################################################################



def create_tfrecord_155_test(batch, filenames0):

    dataset = tf.data.TFRecordDataset(filenames0)
    dataset = dataset.shuffle(buffer_size=(len(filenames0)))
    dataset = dataset.repeat()
    dataset = dataset.prefetch(int(len(filenames0)/batch))
    dataset = dataset.batch(batch_size=len(filenames0))
    dataset = dataset.map(map_func=parse_fn_test, num_parallel_calls=32)
    iter = dataset.make_one_shot_iterator()
    image, label = iter.get_next()
    image = tf.reshape(image, shape=(-1, 50, 60, 60))
    label = tf.one_hot(label, 155)

    return image, label


def create_tfrecord_155_testVote(batch, filenames0):

    dataset = tf.data.TFRecordDataset(filenames0)
    dataset = dataset.shuffle(buffer_size=(len(filenames0)))
    dataset = dataset.repeat()
    dataset = dataset.prefetch(int(len(filenames0)/batch))
    dataset = dataset.batch(batch_size=len(filenames0))
    dataset = dataset.map(map_func=parse_fn_testVote, num_parallel_calls=32)
    iter = dataset.make_one_shot_iterator()
    image, label, videoId = iter.get_next()
    image = tf.reshape(image, shape=(-1, 50, 60, 60))
    label = tf.one_hot(label, 155)

    return image, label, videoId


