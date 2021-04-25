"""Image analysis using pre-trained neural nets

This script runs inference of the various neural nets specified
below. Input images for the nets are loaded from the path specified
by command line argument --image_path. Outputs of each net are stored
separately as text files in the results directory. If --visualize
command line argument is True, nets that support visual output will
store it in the ./result/visualize directory.

To run a forward pass for a specific neural net, the run() function of
each net is imported as run_net_name from net`s script.

This script contains following functions:

    * get_arg - parses command line arguments
"""

import argparse
# Time measurements are being used for testing purposes only
import time

from NSFW_detection import run as run_nsfw_detect
from scene_description import run as run_scene_description
from age_gender import run as run_age_gender_detect
from object_detect import run as run_object_detect


def get_args():
    parser = argparse.ArgumentParser(
        description='Image analysis using pre-trained neural networks.')
    parser.add_argument("--image_path", type=str, default='img/faces2.jpg',
                        help="Target image or directory, if directory script"
                             "will run on all images within it, default img.")
    parser.add_argument("--visualize", type=bool, default=True,
                        help="If True object detection and age_gender "
                             "estimation would be drawn on each image, "
                             "default False.")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    image_path = get_args().image_path
    visualize = get_args().visualize

    start = time.time()
    run_object_detect(image_path, visualize)
    tf2_obj_detect_time = time.time()
    run_nsfw_detect(image_path)
    nsfw_time = time.time()
    run_scene_description(image_path)
    scene_desc_time = time.time()
    run_age_gender_detect(image_path, visualize)
    end = time.time()
    print(f"[INFO] TF2 object detection took "
          f"{tf2_obj_detect_time - start:.2f} seconds")
    print(f"[INFO] NSFW classification took "
          f"{nsfw_time - tf2_obj_detect_time:.2f} seconds")
    print(f"[INFO] Scene classification took "
          f"{scene_desc_time - nsfw_time:.2f} seconds")
    print(f"[INFO] Age-gender classification took "
          f"{end - scene_desc_time:.2f} seconds")
    print(f"[INFO] Total classification took {end - start:.2f} seconds")
