import cv2
import glob
import numpy as np
import random
from random import shuffle


PATH_TO_IMAGES_DIR = '/CASIA-B/'                                        # Path to the original images of the dataset

PATH_TO_SIL_DIR = '/CASIA-B_silhouettes/'                               # Path to the calculated silhouettes

OUTPUT_PATH = "/MulPerGait_two_persons/"                                # Output path

# Dataset metadata
height = 240
width = 320
frame_rate = 25.0


# Augmentation configuration. Each subject have 6 videos, one point of view (90º) and any video is merged with all foreground videos 1 time.
videos_per_subject = 6
points_of_view = 1
videos_to_merge = 1


np.random.seed(0)


def generate_image(image_base, image_silhouettes, mask_silhouettes):
    _, mask_silhouettes = cv2.threshold(mask_silhouettes, 127, 1, cv2.THRESH_BINARY)
    image_silhouettes[:, :, 0] *= mask_silhouettes
    image_silhouettes[:, :, 1] *= mask_silhouettes
    image_silhouettes[:, :, 2] *= mask_silhouettes

    image_base[:, :, 0] *= 1 - mask_silhouettes
    image_base[:, :, 1] *= 1 - mask_silhouettes
    image_base[:, :, 2] *= 1 - mask_silhouettes

    return image_base + image_silhouettes


ids = list(range(75, 125))                                              # Id of users used to generate videos

perm_ids_1 = np.random.permutation(ids)


for i in range(len(perm_ids_1)):
    paths_subject_1 = sorted([f.replace(PATH_TO_IMAGES_DIR, "") for f in glob.glob(
        PATH_TO_IMAGES_DIR + str(perm_ids_1[i]).zfill(3) + "*-090.avi",                     # In our case, we only use the 90º videos
        recursive=True)])

    paths_subject_1 = [filtro_nombres for filtro_nombres in paths_subject_1 if not ('nm-01' in filtro_nombres or 'nm-02' in filtro_nombres or 'nm-03' in filtro_nombres or 'nm-04' in filtro_nombres)]  # We discard the training videos (fine-tuning)

    paths_subject_1 = np.asarray(paths_subject_1)
    paths_subject_1 = paths_subject_1.reshape(videos_per_subject, points_of_view)

    uniques_paths_subject_1_2 = []
    uniques_paths_subject_1_3 = []
    for l in range(videos_per_subject):
        uniques_paths_subject_1_2.append(random.choice(paths_subject_1[l]))
        uniques_paths_subject_1_3.append(random.choice(paths_subject_1[l]))


    for j in range(videos_per_subject):
        perm_ids_2 = np.random.permutation(ids)
        perm_ids_3 = np.random.permutation(ids)

        offset_ids_2 = 0
        offset_ids_3 = 0

        if (perm_ids_1[i] == perm_ids_2[0:videos_to_merge]).any():
            offset_ids_2 = offset_ids_2 + videos_to_merge
            if (perm_ids_2[offset_ids_2:offset_ids_2+videos_to_merge] == perm_ids_3[offset_ids_3:offset_ids_3+videos_to_merge]).any():
                offset_ids_2 = offset_ids_2 + videos_to_merge

        if (perm_ids_1[i] == perm_ids_3[0:videos_to_merge]).any():
            offset_ids_3 = offset_ids_3 + videos_to_merge
            if (perm_ids_2[offset_ids_2:offset_ids_2+videos_to_merge] == perm_ids_3[offset_ids_3:offset_ids_3+videos_to_merge]).any():
                offset_ids_2 = offset_ids_2 + videos_to_merge

        if (perm_ids_2[offset_ids_2:offset_ids_2+videos_to_merge] == perm_ids_3[offset_ids_3:offset_ids_3+videos_to_merge]).any():
            offset_ids_2 = offset_ids_2 + videos_to_merge


        name_2 = ""
        name_3 = ""

        paths_temp_2 = []
        paths_temp_3 = []

        for l in range(videos_to_merge):
            art_vid = ["nm-05", "nm-06"]                        # Foreground videos
            shuffle(art_vid)
            paths_subject_2 = sorted([f.replace(PATH_TO_IMAGES_DIR, "") for f in glob.glob(
                PATH_TO_IMAGES_DIR + str(perm_ids_2[offset_ids_2+l]).zfill(3) + "-" + art_vid[0] + "-090.avi",
                recursive=True)])

            paths_subject_3 = sorted([f.replace(PATH_TO_IMAGES_DIR, "") for f in glob.glob(
                PATH_TO_IMAGES_DIR + str(perm_ids_3[offset_ids_3+l]).zfill(3) + "-" + art_vid[1] + "-090.avi",
                recursive=True)])

            paths_temp_2.append(random.choice(paths_subject_2))
            paths_temp_3.append(random.choice(paths_subject_3))

            name_2 = name_2 + "_" + paths_temp_2[-1][:-4]
            name_3 = name_3 + "_" + paths_temp_3[-1][:-4]



        out2 = cv2.VideoWriter(
            OUTPUT_PATH + uniques_paths_subject_1_2[j][:-4] + name_2 +
            ".mp4", cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))

        out3 = cv2.VideoWriter(
            OUTPUT_PATH + uniques_paths_subject_1_3[j][:-4] + name_3 +
            "_M.mp4", cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))

        images1_2 = []
        images1_3 = []

        vidcap = cv2.VideoCapture(PATH_TO_IMAGES_DIR+uniques_paths_subject_1_2[j])
        success, image = vidcap.read()
        while success:
            images1_2.append(image)
            success, image = vidcap.read()

        vidcap = cv2.VideoCapture(PATH_TO_IMAGES_DIR+uniques_paths_subject_1_3[j])
        success, image = vidcap.read()
        while success:
            images1_3.append(image)
            success, image = vidcap.read()


        for l in range(videos_to_merge):

            images2 = []
            images3 = []

            vidcap = cv2.VideoCapture(PATH_TO_IMAGES_DIR+paths_temp_2[l])
            success, image = vidcap.read()
            while success:
                images2.append(image)
                success, image = vidcap.read()

            vidcap = cv2.VideoCapture(PATH_TO_IMAGES_DIR+paths_temp_3[l])
            success, image = vidcap.read()
            while success:
                images3.append(image)
                success, image = vidcap.read()

            for k in range(len(images1_2)):

                image_base2 = images1_2[k].copy()
                # With subject 2
                if k < len(images2):
                    image_silhouettes = images2[k]
                    mask_silhouettes = cv2.imread(PATH_TO_SIL_DIR+ paths_temp_2[l][:-4] + "/" + str(k).zfill(3) + ".jpg", 0)
                    if np.all(mask_silhouettes != None):
                        mask_silhouettes = cv2.resize(mask_silhouettes, (width, height), interpolation = cv2.INTER_AREA)
                        added_image = generate_image(image_base2, image_silhouettes, mask_silhouettes)
                    else:
                        added_image = image_base2
                else:
                    added_image = image_base2
                images1_2[k] = added_image

            for k in range(len(images1_3)):
                image_base3 = images1_3[k].copy()
                # With subject 3
                if k < len(images3):
                    image_silhouettes = images3[k]
                    mask_silhouettes = cv2.imread(
                        PATH_TO_SIL_DIR + paths_temp_3[l][:-4] + "/" + str(k).zfill(3) + ".jpg", 0)
                    if np.all(mask_silhouettes != None):
                        mask_silhouettes = cv2.resize(mask_silhouettes, (width, height), interpolation=cv2.INTER_AREA)
                        added_image = generate_image(image_base3, cv2.flip(image_silhouettes, 1 ), cv2.flip(mask_silhouettes, 1 ))
                    else:
                        added_image = image_base3
                else:
                    added_image = image_base3
                images1_3[k] = added_image

        for k in range(len(images1_2)):
            out2.write(images1_2[k])

        for k in range(len(images1_3)):
            out3.write(images1_3[k])

        out2.release()
        out3.release()
