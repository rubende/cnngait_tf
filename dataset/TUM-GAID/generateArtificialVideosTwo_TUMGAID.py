import cv2
import glob
import numpy as np


PATH_TO_IMAGES_DIR = '/TUM_GAID/image/'                                 # Path to the original images of the dataset

PATH_TO_SIL_DIR = '/TUM_GAID/silhouettes/'                              # Path to the calculated silhouettes

PATH_ID_FILE = '/TUM_GAID/tumgaidtestids.lst'                           # Path to id list. Used by us to build the test dataset with the indicated users

OUTPUT_PATH = "/MulPerGait_two_persons/"                                # Output path

VIDEOS_BASE = ["b01", "b02", "n05", "n06", "s01", "s02"]                # We limit the types of video to use to maintain the same evaluation process
VIDEOS_SIL = ["n05", "n06"]


np.random.seed(0)

# Function to merge two images using an base image, a second image to merge and its mask silhouette
def generate_image(image_base, image_silhouettes, mask_silhouettes):
    _, mask_silhouettes = cv2.threshold(mask_silhouettes, 127, 1, cv2.THRESH_BINARY)
    image_silhouettes[:, :, 0] *= mask_silhouettes
    image_silhouettes[:, :, 1] *= mask_silhouettes
    image_silhouettes[:, :, 2] *= mask_silhouettes

    image_base[:, :, 0] *= 1 - mask_silhouettes
    image_base[:, :, 1] *= 1 - mask_silhouettes
    image_base[:, :, 2] *= 1 - mask_silhouettes

    return image_base + image_silhouettes


ids = np.loadtxt(PATH_ID_FILE).astype(int)                               # Id of users used to generate videos

perm_ids_1 = np.random.permutation(ids)


for i in range(len(perm_ids_1)):
    for j in VIDEOS_BASE:
        perm_ids_2 = np.random.permutation(ids)
        perm_ids_3 = np.random.permutation(ids)

        offset_ids_2 = 0
        offset_ids_3 = 0

        if perm_ids_1[i] == perm_ids_2[0]:
            offset_ids_2 = offset_ids_2 + 1

        if perm_ids_1[i] == perm_ids_3[0]:
            offset_ids_3 = offset_ids_3 + 1

        if perm_ids_2[0 + offset_ids_2] == perm_ids_3[0 + offset_ids_3]:
            offset_ids_2 = offset_ids_2 + 1

        paths_subject_1 = sorted([f.replace(PATH_TO_IMAGES_DIR, "") for f in glob.glob(
            PATH_TO_IMAGES_DIR + "p" + str(perm_ids_1[i]).zfill(3) + "/" + j + "/*.jpg",
            recursive=True)])

        paths_subject_2 = sorted([f.replace(PATH_TO_IMAGES_DIR, "") for f in glob.glob(
            PATH_TO_IMAGES_DIR + "p" + str(perm_ids_2[0 + offset_ids_2]).zfill(3) + "/" + VIDEOS_SIL[0] + "/*.jpg",
            recursive=True)])

        paths_subject_3 = sorted([f.replace(PATH_TO_IMAGES_DIR, "") for f in glob.glob(
            PATH_TO_IMAGES_DIR + "p" + str(perm_ids_3[0 + offset_ids_3]).zfill(3) + "/" + VIDEOS_SIL[1] + "/*.jpg",
            recursive=True)])

        out2 = cv2.VideoWriter(
            OUTPUT_PATH + paths_subject_1[0][:4] + paths_subject_1[0][5:8] + "_" + paths_subject_2[0][:4] +
                    paths_subject_2[0][5:8] +  ".mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (640, 480))

        out3 = cv2.VideoWriter(
            OUTPUT_PATH + paths_subject_1[0][:4] + paths_subject_1[0][5:8] + "_" + paths_subject_3[0][:4] +
            paths_subject_3[0][5:8] + ".mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (640, 480))


        for k in range(len(paths_subject_1)):

            image_base1 = cv2.imread(PATH_TO_IMAGES_DIR + paths_subject_1[k], 1)
            image_base2 = image_base1.copy()

            # With subject 2
            if k < len(paths_subject_2):
                image_silhouettes = cv2.imread(PATH_TO_IMAGES_DIR + paths_subject_2[k], 1)
                mask_silhouettes =  cv2.imread(PATH_TO_SIL_DIR+paths_subject_2[k], 0)
                mask_silhouettes = cv2.resize(mask_silhouettes, (640, 480), interpolation = cv2.INTER_AREA)
                added_image = generate_image(image_base1, image_silhouettes, mask_silhouettes)
            else:
                added_image = image_base1

            out2.write(added_image)

            # With subject 3
            if k < len(paths_subject_3):
                image_silhouettes = cv2.imread(PATH_TO_IMAGES_DIR + paths_subject_3[k], 1)
                mask_silhouettes = cv2.imread(PATH_TO_SIL_DIR + paths_subject_3[k], 0)
                mask_silhouettes = cv2.resize(mask_silhouettes, (640, 480), interpolation=cv2.INTER_AREA)
                added_image = generate_image(image_base2, image_silhouettes, mask_silhouettes)
            else:
                added_image = image_base2

            out3.write(added_image)

        out2.release()
        out3.release()
