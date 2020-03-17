import cv2
import glob
import numpy as np


PATH_TO_IMAGES_DIR = '/TUM_GAID/image/'                                 # Path to the original images of the dataset

PATH_TO_SIL_DIR = '/TUM_GAID/silhouettes/'                              # Path to the calculated silhouettes

PATH_ID_FILE = '/TUM_GAID/tumgaidtestids.lst'                           # Path to id list. Used by us to build the test dataset with the indicated users

OUTPUT_PATH = "/MulPerGait_one_person/"                                 # Output path

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
    for j in VIDEOS_SIL:

        paths_subject_1 = sorted([f.replace(PATH_TO_IMAGES_DIR, "") for f in glob.glob(
            PATH_TO_IMAGES_DIR + "p" + str(perm_ids_1[i]).zfill(3) + "/" + j + "/*.jpg",
            recursive=True)])


        out = cv2.VideoWriter(
            OUTPUT_PATH + paths_subject_1[0][:4] + paths_subject_1[0][5:8] +  ".mp4",
            cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (640, 480))


        for k in range(len(paths_subject_1)):

            image_base1 = cv2.imread(PATH_TO_IMAGES_DIR + "p" + str(perm_ids_1[i]).zfill(3)+ "/back/background.jpg", 1)         # We store a background image for each subject
            image_silhouettes = cv2.imread(PATH_TO_IMAGES_DIR + paths_subject_1[k], 1)
            mask_silhouettes = cv2.imread(PATH_TO_SIL_DIR + paths_subject_1[k], 0)
            mask_silhouettes = cv2.resize(mask_silhouettes, (640, 480), interpolation=cv2.INTER_AREA)
            added_image = generate_image(image_base1, image_silhouettes, mask_silhouettes)

            out.write(added_image)

        out.release()
