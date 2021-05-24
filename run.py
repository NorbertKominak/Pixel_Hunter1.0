"""Image analysis using pre-trained neural nets and external APIs

This script executes image analysis using 4 neural networks, each
specialized in different task, and APIs` models related to the
networks, if allowed. Currently supported APIs are Amazon Rekognition,
Google Cloud Vision and Microsoft Computer Vision. Each API requires
credentials to be provided. More info on setting up APIs in README.md.

Input images are loaded from the directory specified by input_dir
argument. All outputs are stored in the directory specified by
output_dir. Whether APIs are allowed to run or not is defined by
allow_api argument. Only selected images are sent for further analysis
to APIs. The selection is determined by each neural network`s outputs.

To run a forward pass for a specific neural net and APIs` models
related to it, the run() function of each net`s module is imported
as run_net_name.

This script contains following functions:

    * get_arg - parses command line arguments
"""

import argparse
import os

# To silence TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from NSFW_detection import run as run_nsfw_detect
from scene_classification import run as run_scene_classification
from age_gender import run as run_age_gender_detect
from object_detect import run as run_object_detect


def get_args():
    parser = argparse.ArgumentParser(
        description='Image analysis using pre-trained neural networks'
                    ' and external APIs')
    parser.add_argument("--input_dir", type=str, default='img',
                        help="Input directory. The script will run analysis"
                             "on images within it, but will not search in"
                             "its subdirectories, default=img")
    parser.add_argument("--output_dir", type=str, default='outputs',
                        help="Output directory. All analysis outputs"
                             "will be stored there. All previous results "
                             "will be rewritten. Default=outputs.")
    parser.add_argument("--allow_api", type=bool, default=False,
                        help="If True, some images will be sent to the"
                             "APIs for further analysis, default=False.")

    args = parser.parse_args()
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Allow api: {args.allow_api}\n")
    if not os.path.exists(args.input_dir):
        print("[Error] Invalid input_dir")
        return None

    if not os.path.exists(args.output_dir):
        print("[Error] Invalid output_dir")
        return None

    return args


args = get_args()
if args is not None:
    run_object_detect(args.input_dir, args.output_dir, args.allow_api)
    run_nsfw_detect(args.input_dir, args.output_dir, args.allow_api)
    run_scene_classification(args.input_dir, args.output_dir, args.allow_api)
    run_age_gender_detect(args.input_dir, args.output_dir, args.allow_api)
