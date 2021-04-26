"""NSFW Detection Model# Path to the pre-trained caffe model file

This module provides functionality to run inference of the
NSFW Detection Model using the run() function.

This module contains following functions:

    * to_output_file - post processes classification result to txt file
    * run - runs inference on an image/images and stores results

"""

from nsfw_detector import predict
from typing import Dict

# Path to the pre-trained model file
MODEL_PATH = "models/nsfw.299x299.h5"

# Required image input size
NETWORK_INPUT_SIZE = 299


def to_output_file(classification: Dict[str, Dict[str, float]]) -> None:
    """Post processes results of the NSFW classification and store them
    in a text file located at the results directory.

    Parameters
    ----------
    classification : Dict[str, Dict[str, float]]
        Results of the NSFW classification. Keys represent path to an
        image and values represent dictionary with labels and their
        scores.

    """
    with open("results/NSFW_detection.txt", "w") as output_file:
        for img_path, categories in classification.items():
            img_path = img_path[img_path.rfind('\\') + 1:]
            result = f'{img_path};\n'
            for category, value in categories.items():
                result += f'{category}: {(value * 100):.2f}%; '
            result += '\n'

            output_file.write(result)


def run(img_path: str) -> None:
    """ Runs inference of the Scene Description Model on images located
    at img_path. Stores results in a text file located in the results
    directory, using to_output_file() function.

    Parameters
    ----------
    img_path : str
        Path to an image or dir. If dir, then runs inference on all
        images located within the dir.

    """
    model = predict.load_model(MODEL_PATH)
    nsfw_classification = predict.classify(model, img_path,
                                           image_dim=NETWORK_INPUT_SIZE)
    to_output_file(nsfw_classification)
