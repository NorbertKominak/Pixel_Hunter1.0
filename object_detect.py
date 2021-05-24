"""Object Detection Module

This module provides functionality to run inference with most of object
detection models from TensorFlow 2 Detection Model Zoo
(https://github.com/tensorflow/models/blob/master/research
/object_detection/g3doc/tf2_detection_zoo.md) using the run() function.
If allowed it also sends selected images to the object detection models
of the Amazon Rekognition, Google Vision Cloud and Microsoft Computer
Vision APIs.


This module contains following functions:

    * load_labels - loads labels of the MSCOCO 2017 dataset
    * run - runs inference on images and stores results

"""

import cv2
import numpy as np
import os
import tensorflow as tf
from awsvision import run_aws_api
from msvision import run_ms_api
from ggvision import run_gg_api

# Path to the saved_model dir containing variables and TensorFlows`s
# protobuf file containing networks definition and weights.
MODEL_PATH = "models/efficientdet_d1_coco17_tpu-32/saved_model"

# Required input image size for the network
NETWORK_INPUT_SIZE = 640

# Path to the MSCOCO 2017 labels path
LABELS_PATH = "labels/mscoco_label_map.pbtxt"

# Threshold determining the minimal confidence for the bounding box
# to be counted as prediction
CONFIDENCE_THRESHOLD = 0.4

# Threshold determining whether prediction`s confidence is trustworthy
# or will be sent to APIs for further analysis
API_THRESHOLD = 0.5


def load_labels(labels_path):
    """Loads labels of the MSCOCO 2017 dataset

    Parameters
    ----------
    labels_path : str
        The file location of the mscoco_label_map.

    Returns
    -------
        Dictionary where key corresponds to the label`s index
        in the output layer and value to the label`s name.

    """
    labels = {}
    with open(labels_path) as labels_file:
        class_id = -1
        for line in labels_file:
            if line.find('id') != -1:
                class_id = int(line.split(':')[1].strip())
                labels[class_id] = ''
                continue

            if line.find('display_name') != -1:
                labels[class_id] = line.split(':')[1].strip().replace('"', '')

    return labels


def run(input_dir, output_dir, allow_api):
    """ Runs inference of the Object Detection model on images located
    in the input_dir. Stores results in a .csv file located in the
    output_dir. If allowed, run() sends selected images to object
    detection models of the Amazon Rekognition, Google Vision Cloud
    and Microsoft Computer Vision APIs. Only images with any
    prediction`s score lower than the API_THRESHOLD constant are sent
    to APIs. APIs` results are also stored in the output_dir.

    Parameters
    ----------
    input_dir : str
        Path to the dir with images to run inference on.
    output_dir : str
        Path to the dir where all results will be stored.
    allow_api : bool
        If True selected images are sent to the APIs.

    """
    labels = load_labels(LABELS_PATH)
    imgs_for_api = set()
    count_img = 0

    print("[OBJECT DETECTION] Loading model...")
    print("[OBJECT DETECTION] You can ignore the following warnings "
          "regarding custom gradient")
    obj_detect_net = tf.saved_model.load(MODEL_PATH)

    with open(os.path.join(output_dir, "det_object.csv"), "w") as output_file:
        output_file.write("img_name,label_name,score,left,top,right,bottom\n")
        for file in os.listdir(input_dir):
            img = cv2.imread(os.path.join(input_dir, file))
            if img is not None:
                count_img += 1
                # Expand dimensions since the model expects images to
                # have shape: [1, None, None, 3]
                input_tensor = np.expand_dims(
                    cv2.resize(img,
                               (NETWORK_INPUT_SIZE, NETWORK_INPUT_SIZE)), 0)

                detections = obj_detect_net(input_tensor)

                # Indices of detected classes
                bclasses = detections['detection_classes'][0].numpy(). \
                    astype(np.int32)

                # Scores of detected classes
                bscores = detections['detection_scores'][0].numpy()

                # Coordinates of boxes related to the detected class
                bboxes = detections['detection_boxes'][0].numpy()

                none_detection = True
                for idx in range(len(bscores)):
                    if bscores[idx] >= CONFIDENCE_THRESHOLD:

                        none_detection = False
                        # Stores images that will be sent for further analysis
                        if allow_api and bscores[idx] < API_THRESHOLD:
                            imgs_for_api.add(file)

                        im_height, im_width, _ = img.shape
                        y_min = int(bboxes[idx][0] * im_height)
                        x_min = int(bboxes[idx][1] * im_width)
                        y_max = int(bboxes[idx][2] * im_height)
                        x_max = int(bboxes[idx][3] * im_width)

                        output = f"{file},{labels[bclasses[idx]]}," \
                                 f"{bscores[idx]:.2f},{x_min},{y_min}," \
                                 f"{x_max},{y_max}\n"
                        output_file.write(output)

                if none_detection:
                    output_file.write(f"{file}\n")



    print(f"[OBJECT DETECTION] Analyzed {count_img} images, results are "
          f"stored in {os.path.join(output_dir, 'det_object.csv')}")
    print(f"[OBJECT DETECTION] Passing {len(imgs_for_api)} images for "
          "further analysis to APIs...")

    run_aws_api(imgs_for_api, input_dir, output_dir, "detect_objects")
    run_ms_api(imgs_for_api, input_dir, output_dir, "detect_objects")
    run_gg_api(imgs_for_api, input_dir, output_dir, "detect_objects")
    print("-------------------------------------------------------\n")
