import cv2
import os
import numpy as np


PATH_TO_IMAGES_DIR = '/CASIA-B/'                                        # Path to the original images of the dataset

PATH_TO_SIL_DIR = '/CASIA-B_silhouettes/'                               # Path to the calculated silhouettes

OUTPUT_PATH = "/MulPerGait_one_person/"                                 # Output path

VIDEOS_SIL = ["nm-05-090", "nm-06-090"]                                 # We limit the types of video to use to maintain the same evaluation process
VIDEOS_BASE_PATH = "CASIA-B_background.png"

# Dataset metadata
height = 240 
width = 320
frame_rate = 25.0

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
    for j in VIDEOS_SIL:

        vidcap = cv2.VideoCapture(PATH_TO_IMAGES_DIR + str(perm_ids_1[i]).zfill(3) + "-" + j + ".avi")


        out = cv2.VideoWriter(
            OUTPUT_PATH + str(perm_ids_1[i]).zfill(3) + "-" + j + ".mp4",
            cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))


        k = 0
        success, image = vidcap.read()
        while success:

            if not os.path.isfile(PATH_TO_SIL_DIR + str(perm_ids_1[i]).zfill(3) + "-" + j + "/" + str(k).zfill(3) + ".jpg"):
                added_image = cv2.imread(VIDEOS_BASE_PATH, 1)
            else:
                image_base1 = cv2.imread(VIDEOS_BASE_PATH, 1)
                image_silhouettes = image
                mask_silhouettes = cv2.imread(PATH_TO_SIL_DIR + str(perm_ids_1[i]).zfill(3) + "-" + j + "/" + str(k).zfill(3) + ".jpg", 0)
                mask_silhouettes = cv2.resize(mask_silhouettes, (width, height), interpolation=cv2.INTER_AREA)
                added_image = generate_image(image_base1, image_silhouettes, mask_silhouettes)

            out.write(added_image)
            success, image = vidcap.read()
            k = k + 1

        out.release()
