"""Object detection

This module provides functionality to run inference of any object
detection model from TensorFlow 2 Detection Model Zoo
(https://github.com/tensorflow/models/blob/master/research
/object_detection/g3doc/tf2_detection_zoo.md) using the run() function.

This module contains following functions:

    * load_labels - loads labels of the MSCOCO 2017 dataset
    * draw_rectangle - draws a rectangle with a label around
                       a detected object
    * run - runs inference on an image/images and stores results

"""

import cv2
import numpy as np
import tensorflow as tf
import helpers
from typing import List, Dict

# Path to the saved_model dir containing variables and TensorFlows`s
# protobuf file containing networks definition and weights.
MODEL_PATH = "models/efficientdet_d1_coco17_tpu-32/saved_model"

# Required input image size for the network
NETWORK_INPUT_SIZE = 640

# Path to the MSCOCO 2017 labels path
LABELS_PATH = "labels/mscoco_label_map.pbtxt"


def load_mscoco_labels(labels_path: str) -> Dict[int, str]:
    """Loads labels of the MSCOCO 2017 dataset

    Parameters
    ----------
    labels_path : str
        The file location of the mscoco_label_map.

    Returns
    -------
    dict[int, str]
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


def draw_rectangle(img: np.ndarray, bbox: List[float], label: str) \
        -> None:
    """Draws a rectangle with a label around detected object.

    Parameters
    ----------
    img : np.ndarray
        Image representation as an numpy array.
    bbox : List[float]
        Array with coordinates of the bounding box. The expected order
        is: y_min, x_min, y_max, x_max.
    label : str
        Label of the detected object.

    """
    im_height, im_width, _ = img.shape
    y_min = int(bbox[0] * im_height)
    x_min = int(bbox[1] * im_width)
    y_max = int(bbox[2] * im_height)
    x_max = int(bbox[3] * im_width)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.putText(img, label, (x_min + 5, y_min - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)


def run(img_path: str, visualize: bool = False) -> None:
    """ Runs inference of the object detection model on images located
    at img_path. Stores results in a text file located in the results
    directory. If visualize is True, stores visualized results in the
    results/visualize/object_detect directory.

    Parameters
    ----------
    img_path : str
        Path to an image or dir. If dir, then runs inference on all
        images located within the dir.
    visualize : bool, optional
        Flag whether visualized result should be stored as well.
        Default False.

    """
    labels = load_mscoco_labels(LABELS_PATH)

    # tf.keras.backend.clear_session()
    obj_detect_net = tf.saved_model.load(MODEL_PATH)

    image_generator = helpers.yield_images_from_dir(img_path,
                                                    NETWORK_INPUT_SIZE)
    with open("results/object_detect.txt", "w") as output_file:
        for img, img_name in image_generator:
            result = f'{img_name};\n'
            n = 0

            # Expand dimensions since the model expects images to
            # have shape: [1, None, None, 3]
            input_tensor = np.expand_dims(img, 0)
            detections = obj_detect_net(input_tensor)

            # Indices of detected classes
            bclasses = detections['detection_classes'][0].numpy().\
                astype(np.int32)

            # Scores of detected classes
            bscores = detections['detection_scores'][0].numpy()

            if visualize:
                # Coordinates of boxes related to the detected class
                bboxes = detections['detection_boxes'][0].numpy()

            for idx in range(len(bscores)):
                if bscores[idx] >= 0.4:
                    n += 1
                    label = f'  {labels[bclasses[idx]]} : ' \
                            f'{bscores[idx] * 100:.2f}%'

                    if visualize:
                        bbox = [bboxes[idx][0], bboxes[idx][1],
                                bboxes[idx][2], bboxes[idx][3]]

                        draw_rectangle(img, bbox, label)
                        cv2.imwrite(f'results/visualize/object_detect/'
                                    f'{img_name}', img)

                    result += label + "\n"
                    result += f'    Amount of objects detected: {n}\n'

            output_file.write(result)
