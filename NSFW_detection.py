"""NSFW Detection Module

This module provides functionality to run inference of the NSFW
Detection Model using the run() function. If allowed it also sends
selected images to content moderation models of the Amazon Rekognition,
Google Vision Cloud and Microsoft Computer Vision APIs.

This module contains following functions:

    * run - runs inference on an image/images and stores results

"""

import os
from nsfw_detector import predict
from awsvision import run_aws_api
from msvision import run_ms_api
from ggvision import run_gg_api

# Path to the pre-trained model file
MODEL_PATH = "models/nsfw.299x299.h5"

# Required image input size
NETWORK_INPUT_SIZE = 299

# If the top prediction`s score is below this threshold the image will
# be sent to APIs for further analysis
API_THRESHOLD = 0.9


def run(input_dir, output_dir, allow_api):
    """ Runs inference of the Scene Description Model on images located
    in the input_dir. Stores results in a .csv file located in the
    output_dir. If allowed, run() sends selected images to content
    moderation models of the Amazon Rekognition, Google Vision Cloud
    and Microsoft Computer Vision APIs. Images with the top
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
    print("[NSFW DETECTION] Loading model...")
    model = predict.load_model(MODEL_PATH)
    nsfw_classification = dict()
    imgs_for_api = set()

    try:
        nsfw_classification = predict.classify(model, input_dir,
                                               image_dim=NETWORK_INPUT_SIZE)

    except ValueError:
        print("[Error] Input directory does not contain any valid images")

    with open(os.path.join(output_dir, "moderate_image.csv"), "w") \
            as output_file:

        for img in nsfw_classification.keys():
            img_name = img[img.rfind('\\') + 1:]
            if (allow_api and
                    max(nsfw_classification[img].values()) < API_THRESHOLD):
                imgs_for_api.add(img_name)

            img_name = img[img.rfind('\\') + 1:]
            output = f"{img_name}"
            for cat, score in nsfw_classification[img].items():
                output += f",{cat}:{score:.2f}"

            output += "\n"
            output_file.write(output)

    print(f"[NSFW DETECTION] Analyzed {len(nsfw_classification)} images, "
          "results are stored in "
          f"{os.path.join(output_dir, 'moderate_image.csv')}")

    print(f"[NSFW DETECTION] Passing {len(imgs_for_api)} images for"
          " further analysis to APIs...")

    run_aws_api(imgs_for_api, input_dir, output_dir, "moderate_image")
    run_ms_api(imgs_for_api, input_dir, output_dir, "moderate_image")
    run_gg_api(imgs_for_api, input_dir, output_dir, "moderate_image")
    print("-------------------------------------------------------\n")
