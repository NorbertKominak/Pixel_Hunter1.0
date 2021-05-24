"""Scene Classification Module

This module provides functionality to run inference of the
Scene Classification Model using the run() function. If allowed it
also sends selected images to scene classification models of the Amazon
Rekognition, Google Vision Cloud and Microsoft Computer Vision APIs.

This module contains following functions:

    * load_labels - loads labels of the Standard365 Places2 Dataset
    * run - runs inference on an image/images and stores results

"""

import cv2
import numpy as np
import os
from awsvision import run_aws_api
from msvision import run_ms_api
from ggvision import run_gg_api

# Path to the pre-trained caffe model file
MODEL_PATH = "models/resnet152_places365.caffemodel"

# Path to the pre-trained caffe model configuration file
PROTO_TXT_PATH = "models/deploy_resnet152_places365.prototxt"

# Required network`s input image size
NETWORK_INPUT_SIZE = 224

# Path to the file containing labels from the Standard365
# Places2 Dataset
LABELS_PATH = "labels/IO_places365.txt"

# If the top prediction`s score is below this threshold the image will
# be sent to APIs for further analysis
API_THRESHOLD = 0.4


def load_labels(labels_path):
    """Loads labels of the Standard365 Places2 Dataset

    Parameters
    ----------
    labels_path : str
        The location of the Standard365 Place2 labels file.

    Returns
    -------
        Two lists. First represents class labels. Second
        indoors/outdoors labels matching the first list`s labels.

    """
    classes = []
    outdoors_indoors = []
    with open(labels_path) as labels_file:
        for line in labels_file:
            classes.append(line[3:line.find(' ')])
            outdoors_indoors.append(line[line.find(' '):].strip())

    return classes, outdoors_indoors


def run(input_dir, output_dir, allow_api):
    """ Runs inference of the Scene Classification model on images
    located in the input_dir. Stores results in a .csv file located in
    the output_dir. If allowed, run() sends selected images to scene
    classification models of the Amazon Rekognition, Google Vision
    Cloud and Microsoft Computer Vision APIs. Images with the top
    prediction`s score below the API_THRESHOLD constant are sent to
    APIs. APIs` results are also stored in the output_dir.

    Parameters
    ----------
    input_dir : str
        Path to the dir with images to run inference on.
    output_dir : str
        Path to the dir where all results will be stored.
    allow_api : bool
        If True selected images are sent to the APIs.

    """
    classes, outdoors_indoors = load_labels(LABELS_PATH)
    print("[SCENE CLASSIFICATION] Loading model...")
    scene_classification_net = cv2.dnn.readNetFromCaffe(PROTO_TXT_PATH,
                                                        MODEL_PATH)
    imgs_for_api = set()
    count_img = 0

    with open(os.path.join(output_dir, "scene_classification.csv"), "w") \
            as output_file:

        for file in os.listdir(input_dir):
            img = cv2.imread(os.path.join(input_dir, file))
            if img is not None:
                count_img += 1
                # Preprocess image for inference by doing mean subtraction with
                # values from ImageNet 2012 dataset.
                scene_classification_net.setInput(
                    cv2.dnn.blobFromImage(img, 1,
                                          (NETWORK_INPUT_SIZE,
                                           NETWORK_INPUT_SIZE),
                                          (104, 117, 123)))
                network_output = scene_classification_net.forward()

                # stores indices of top-5 scene classification classes
                idxs = np.argsort(network_output[0])[::-1][:5]

                # send to API
                if allow_api and network_output[0][idxs[0]] < API_THRESHOLD:
                    imgs_for_api.add(file)

                output = f"{file}"
                for (i, idx) in enumerate(idxs):
                    if outdoors_indoors[idx] == "1":
                        out_in = "indoors"
                    else:
                        out_in = "outdoors"

                    output += f",{classes[idx]}:{out_in}:" \
                              f"{network_output[0][idx] * 100:.2f}"

                output += "\n"

                output_file.write(output)

    print(f"[SCENE CLASSIFICATION] Analyzed {count_img} images, results are "
          f"stored in {os.path.join(output_dir, 'scene_classification.csv')}")

    print(f"[SCENE CLASSIFICATION] Passing {len(imgs_for_api)} images for"
          " further analysis to APIs...")

    run_aws_api(imgs_for_api, input_dir, output_dir, "scene_classification")
    run_ms_api(imgs_for_api, input_dir, output_dir, "scene_classification")
    run_gg_api(imgs_for_api, input_dir, output_dir, "scene_classification")
    print("-------------------------------------------------------\n")
