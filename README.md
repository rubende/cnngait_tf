# MuPeG: The Multiple Person Gait framework

Gait recognition on scenes with multiple people.

The paper describing this framework is available [here](https://www.mdpi.com/1424-8220/20/5/1358).

## Getting Started

TO DO.

### Prerequisites

This code needs TensorFlow library to run.

```
pip install tensorflow-gpu
```

### Installing

TO DO

## Create synthetic database

Our synthetic database is built from the TUM-GAID dataset, which has the following file structure:

```
p001
p002
p003
.
.
.
p305
	- b01
	- b02
	- b03
	- b04
	- back
	- back2
	- c01
	- ...
```

If the database to be used does not have the following structure, it will be necessary to make modifications to the codes used to build the synthetic database.


To create a synthetic database from an existing one, the following scripts must be adapted and used:

### objectDetectorSilhouette

Script that calculates and stores the silhouettes of the videos of the original dataset.

It is necessary to modify the following paths within the script:

```
PATH_TO_RESEARCH = '/tensorflow/models/research/'	# Path to /tensorflow/models/research/ installation folder

PATH_TO_TEST_IMAGES_DIR = '/TUM_GAID/image'			# Path to the original images of the dataset

OUTPUT_PATH = '/TUM_GAID/silhouettes/'				# Output path
```

### generateArtificialVideosOne

Generate artificial videos with a single subject using a background image. Background image for each subject is located into subject's __back__ folder.

It is necessary to modify the following paths within the script:

```
PATH_TO_IMAGES_DIR = '/TUM_GAID/image/'				# Path to the original images of the dataset

PATH_TO_SIL_DIR = '/TUM_GAID/silhouettes/'          # Path to the calculated silhouettes

PATH_ID_FILE = '/TUM_GAID/tumgaidtestids.lst'		# Path to id list. Used by us to build the test dataset with the indicated users

OUTPUT_PATH = "/MulPerGait_one_person/"             # Output path
```

### generateArtificialVideosTwo

Generate artificial videos with two subjects. 

It is necessary to modify the following paths within the script:

```
PATH_TO_IMAGES_DIR = '/TUM_GAID/image/'				# Path to the original images of the dataset

PATH_TO_SIL_DIR = '/TUM_GAID/silhouettes/'          # Path to the calculated silhouettes

PATH_ID_FILE = '/TUM_GAID/tumgaidtestids.lst'			# Path to id list. Used by us to build the test dataset with the indicated users

OUTPUT_PATH = "/MulPerGait_two_persons/"                # Output path
```

## Experiments


### objectDetector

Script that calculates and stores the bounding boxes of the subjects in generated videos.

It is necessary to modify the following paths within the script:

```
PATH_TO_TEST_IMAGES_DIR = '/MulPerGait_two_persons/'	# Path to generated dataset

OUTPUT_PATH = '/MulPerGait_two_persons_bb/'             # Output path
```

### cnnTracker

Use the generated videos, its bounding boxes and its optical flow to obtain the tracking information of the subjects.


It is necessary to modify the following paths within the script:

```
PATH_TO_TEST_IMAGES_ORIGINAL_DIR = '/MulPerGait_two_persons/'       	# Path to generated dataset

PATH_TO_OF_DIR = '/MulPerGait_two_persons_of/'         	# Path to optical flow of the generated dataset

PATH_TO_BB_DIR = '/MulPerGait_two_persons_bb/'          # Path to bounding boxes of the generated dataset

OUTPUT_PATH = "/MulPerGait_two_persons_cnn_tr/"         # Output path
```

### generate25Frames

Generates samples windows with 25 frame. Synthetic videos, and their previously calculated optical flows and tracking information are used for this.

It is necessary to modify the following paths within the script:

```
PATH_TO_OF = '/MulPerGait_two_persons_of/'              # Path to optical flow of the generated dataset

PATH_TO_TR = '/MulPerGait_two_persons_cnn_tr/'          # Path to tracking information of the generated dataset

PATH_TO_IMAGE = '/MulPerGait_two_persons/'              # Path to generated dataset

OUTPUT_PATH = '/MulPerGait_two_persons_cnn_25f/'        # Output path
```

### train_150


```
INPUT_PATH = "/inputs_N150/"					# Input data from TUM-GAID dataset. Into /inputs_N150/ we have 3 folders, one for video type. Into this folders, we stored one tfrecord file per video.
OUTPUT_PATH = "/outputs/"						# Output path, where we store the resulting model.
```

### train_155

```
model_150_path = "/outputs/model_150.h5"        # Model_150 path
INPUT_PATH = "/inputs_N155/"                    # Input data from TUM-GAID dataset. Into /inputs_N155/ we have 3 folders, one for video type. Into this folders, we stored one tfrecord file per video.
OUTPUT_PATH = "/outputs/"                       # Output path, where we store the resulting model.
```


### PredictModel_one_person

```
PATH_ID_FILE = "tumgaidtestids.lst"                             # File with the IDs of the users that we are going to use
PATH_25F_INPUT = '/MulPerGait_one_person_cnn_25f/'              # Path with samples windows with 25 frame
PATH_MODEL_CNN = "/outputs/model_155.h5"                        # Model_155 path
```


### PredictModel_two_persons_individual


```
PATH_ID_FILE = "tumgaidtestids.lst"                             # File with the IDs of the users that we are going to use
PATH_25F_INPUT = '/MulPerGait_two_persons_cnn_25f/'             # Path with samples windows with 25 frame
PATH_MODEL_CNN = "/outputs/model_155.h5"                        # Model_155 path
```

### PredictModel_two_persons_all

```
PATH_ID_FILE = "tumgaidtestids.lst"                             # File with the IDs of the users that we are going to use
PATH_25F_INPUT = '/MulPerGait_two_persons_cnn_25f/'             # Path with samples windows with 25 frame
PATH_MODEL_CNN = "/outputs/model_155.h5"                        # Model_155 path
```

## Authors

* **Rubén Delgado Escaño** - [rubende](https://github.com/rubende)
* **Francisco Castro Payán** - [fcastro](https://github.com/fcastro)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Citation

If you use this framework in your research, please cite:

```
@Article{delgado2020mupeg,
  AUTHOR = {Delgado-Escaño, Rubén and Castro, Francisco M. and Cózar, Julián R. and Marín-Jiménez, Manuel J. and Guil, Nicolás},
  TITLE = {MuPeG—The Multiple Person Gait Framework},
  JOURNAL = {Sensors},
  VOLUME = {20},
  YEAR = {2020},
  NUMBER = {5},
  ARTICLE-NUMBER = {1358},
  URL = {https://www.mdpi.com/1424-8220/20/5/1358},
  ISSN = {1424-8220},
  DOI = {10.3390/s20051358}
}
```

## Acknowledgments

* We thank the reviewers for their helpful comments.
