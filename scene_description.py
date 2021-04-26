"""Scene Description Model

This module provides functionality to run inference of the
Scene Description Model using the run() function.

This module contains following functions:

    * load_labels - loads labels of the Standard365 Places2 Dataset
    * run - runs inference on an image/images and stores results

"""

import cv2
import helpers
import numpy as np
from typing import List

# Path to the pre-trained caffe model file
MODEL_PATH = "models/resnet152_places365.caffemodel"

# Path to the pre-trained caffe model configuration file
PROTO_TXT_PATH = "models/deploy_resnet152_places365.prototxt"

# Required network`s input image size
NETWORK_INPUT_SIZE = 224

# Path to the file containing labels from the Standard365
# Places2 Dataset
LABELS_PATH = "labels/IO_places365.txt"


def load_labels(labels_path: str) -> (List[str], List[str]):
    """Loads labels of the Standard365 Places2 Dataset

    Parameters
    ----------
    labels_path : str
        The location of the Standard365 Place2 labels file.

    Returns
    -------
    (list[str], list [str])
        Two lists. First represents class labels. Second
        indoors/outdoors labels matching the first list labels.

    """
    classes = []
    outdoors_indoors = []
    with open(labels_path) as labels_file:
        for line in labels_file:
            classes.append(line[3:line.find(' ')])
            outdoors_indoors.append(line[line.find(' '):].strip())

    return classes, outdoors_indoors


def run(img_path: str) -> None:
    """ Runs inference of the Scene Description Model on images located
    at img_path. Stores results in a text file located in the results
    directory.

    Parameters
    ----------
    img_path : str
        Path to an image or dir. If dir, then runs inference on all
        images located within the dir.

    """
    classes, outdoors_indoors = load_labels(LABELS_PATH)
    scene_classification_net = cv2.dnn.readNetFromCaffe(PROTO_TXT_PATH,
                                                        MODEL_PATH)

    image_generator = helpers.yield_images_from_dir(img_path,
                                                    NETWORK_INPUT_SIZE)
    with open("results/scene_description.txt", "w") as output_file:
        for img, img_name in image_generator:
            # Preprocess image for inference. Does mean subtraction with
            # values from ImageNet 2012 dataset.
            scene_classification_net.setInput(
                cv2.dnn.blobFromImage(img, 1,
                                      (NETWORK_INPUT_SIZE, NETWORK_INPUT_SIZE),
                                      (104, 117, 123)))
            network_output = scene_classification_net.forward()

            # stores indices of top-5 scene classification classes
            idxs = np.argsort(network_output[0])[::-1][:5]
            result = f'{img_name};\n'
            for (i, idx) in enumerate(idxs):
                result += f"Label with probability rank {i + 1}: " \
                          f"{classes[idx]} {network_output[0][idx] * 100:.2f}%"
                out_in = 'indoors' if outdoors_indoors[idx] == 1 else \
                    'outdoors'

                result += f", {out_in}\n"

            output_file.write(result)
