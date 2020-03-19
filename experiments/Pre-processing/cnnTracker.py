import glob
from  keras.applications.resnet50 import ResNet50
import os
import ntpath
import pickle
import cv2
import scipy.io
import numpy as np
from keras.models import Model
from scipy.spatial.distance import pdist


PATH_TO_TEST_IMAGES_ORIGINAL_DIR = '/videos_two_persons/'                   # Path to generated dataset

PATH_TO_OF_DIR = '/videos_two_persons_of/'                                  # Path to optical flow of the generated dataset

PATH_TO_BB_DIR = '/videos_two_persons_bb/'                                  # Path to bounding boxes of the generated dataset

OUTPUT_PATH = "/videos_two_persons_cnn_tr/"                                 # Output path


# Size of the images.
im_height = 480
im_width = 640


TEST_IMAGE_ORIGINAL_PATHS = [f for f in glob.glob(PATH_TO_TEST_IMAGES_ORIGINAL_DIR + "**/*.mp4", recursive=True)]


model = ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.layers[-2].output)

for i in range(len(TEST_IMAGE_ORIGINAL_PATHS)):
    index_video = i
    basename = ntpath.basename(TEST_IMAGE_ORIGINAL_PATHS[index_video])
    basename, file_extension = os.path.splitext(basename)
    output_video = pickle.load(open(PATH_TO_BB_DIR + basename + '.pkl', "rb"))

    f = scipy.io.loadmat(PATH_TO_OF_DIR + "/" + basename + '_ofx10.mat')
    cap = cv2.VideoCapture(TEST_IMAGE_ORIGINAL_PATHS[index_video])
    init_frame = f['S']['frix'][0][0][0][0]

    ids = []
    predictores = []
    position_list = []
    crops_original = []

    for j in range(len(f['S']['frix'][0])):
        frame = f['S']['frix'][0][j][0][0]
        output_dict = output_video[frame - 1]

        cap.set(1, frame - 1)
        ret, image = cap.read()
        pred = []
        post = []
        crops = []
        for k in range(output_dict['num_detections']):
            if output_dict['detection_scores'][k] > 0.95 and output_dict['detection_classes'][k] == 1:
                ymin = int(output_dict['detection_boxes'][k][0] * im_height)
                xmin = int(output_dict['detection_boxes'][k][1] * im_width)
                ymax = int(output_dict['detection_boxes'][k][2] * im_height)
                xmax = int(output_dict['detection_boxes'][k][3] * im_width)
                crop = image[ymin:ymax, xmin:xmax]
                crops.append(crop)
                crop = cv2.resize(crop, (224, 224))
                crop = np.expand_dims(crop, axis=0)
                pred.append(np.squeeze(intermediate_layer_model.predict(crop)))
                post.append((ymin, xmin, ymax, xmax))

        predictores.append(pred/np.linalg.norm(pred))
        position_list.append(post)
        crops_original.append(crops)

    count = 0
    count_id = 1
    for j in range(len(predictores)):
        if len(predictores[j]) == 0:
            ids.append([0])
        if len(predictores[j]) >= 1:
            if count == 0:
                t = list(range(1, len(predictores[j])+1))
                ids.append(t)
                count_id = count_id+len(t)
            else:
                temp_ids = np.zeros(len(predictores[j]))
                dist = pdist(np.append(predictores[j - 1], predictores[j], axis=0))
                dist = dist[sum(range(len(predictores[j-1]))):len(dist) - sum(range(len(predictores[j])))]
                dist = dist.reshape((len(predictores[j-1]), len(predictores[j])))

                if dist.shape[0] > dist.shape[1]:
                    for k in range(len(predictores[j])):
                        t1, t2 = np.unravel_index(dist.argmin(), dist.shape)
                        if dist.min() > 0.5:
                            temp_ids[t2] = count_id
                            count_id = count_id + 1
                        else:
                            temp_ids[t2] = ids[-1][t1]
                        dist[:, t2] = 1
                        dist[t1, :] = 1
                else:
                    for k in range(len(predictores[j-1])):
                        t1, t2 = np.unravel_index(dist.argmin(), dist.shape)
                        if dist.min() > 0.5:
                            temp_ids[t2] = count_id
                            count_id = count_id + 1
                        else:
                            temp_ids[t2] = ids[-1][t1]
                        dist[:, t2] = 1
                        dist[t1, :] = 1

                if len(predictores[j-1]) < len(predictores[j]):
                    for k in range(len(temp_ids)):
                        if temp_ids[k] == 0:
                            temp_ids[k] = count_id
                            count_id = count_id + 1
                ids.append(temp_ids)
        count = len(predictores[j])


    max_count = 0
    previos = []
    previos_pred = []
    pares = None
    for j in range(len(ids)):
        if not np.any(ids[j]):
            continue
        if len(ids[j]) > max_count and np.all(np.isin(previos, ids[j])) or len(ids[j]) > max_count and max_count == 0:
            max_count = len(ids[j])
            previos = ids[j]
            previos_pred = predictores[j]
        else:
            temp1 = [value for value in previos if value not in ids[j]]
            i_temp1 = [ind for ind, x in enumerate(previos) if x not in ids[j]]
            temp2 = [value for value in ids[j] if value not in previos]
            i_temp2 = [ind for ind, x in enumerate(ids[j]) if x not in previos]


            i_temp1 = np.array(i_temp1)
            i_temp2 = np.array(i_temp2)

            dist = pdist(np.squeeze(np.append(previos_pred[i_temp1], predictores[j][i_temp2], axis=0)))
            dist = dist[sum(range(len(previos_pred[i_temp1]))):len(dist) - sum(range(len(predictores[j][i_temp2])))]
            dist = dist.reshape((len(previos_pred[i_temp1]), len(predictores[j][i_temp2])))

            
            for k in range(len(previos_pred[i_temp1])):
                t1, t2 = np.unravel_index(dist.argmin(), dist.shape)
                pares = (ids[j][i_temp2[t2]], np.array(previos)[i_temp1[t1]])
                ids[j][i_temp2[t2]] = np.array(previos)[i_temp1[t1]]
                dist[:, t2] = 1
                dist[t1, :] = 1

        if pares != None:
            for k in range(j+1, len(predictores)):
                flag = False
                for t in range(len(ids[k])):
                    if ids[k][t] == pares[0]:
                        ids[k][t] = pares[1]
                        flag = True
                if flag == False:
                    break
        pares = None
     


    rids = list(range(1, max_count + 1))
    bids = [0] * max_count

    for j in range(len(ids)):
        for k in range(len(ids[j])):
            if ids[j][k] in rids and ids[j][k] != 0:
                bids[int(ids[j][k])-1] = int(ids[j][k])

    for j in range(len(ids)):
        for k in range(len(ids[j])):
            if ids[j][k] not in bids and ids[j][k] != 0:
                for f in range(len(bids)):
                    if bids[f] == 0:
                        bids[f] = int(ids[j][k])
                        ids[j][k] = f+1
            elif ids[j][k] != 0:
                ids[j][k] = bids.index(ids[j][k]) + 1



    with open(OUTPUT_PATH + basename + '.pkl', 'wb') as output:
        pickle.dump([ids, position_list], output, pickle.HIGHEST_PROTOCOL)