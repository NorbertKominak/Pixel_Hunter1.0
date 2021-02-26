import argparse
import time
from object_detection import run as run_object_detect
from NSFW_detection import run as run_NSFW_detect
from scene_description import run as run_scene_description
from age_gender import run as run_age_gender_detect


def get_args():
    parser = argparse.ArgumentParser(description='Run image analysis')
    parser.add_argument("--image_path", type=str, default='img',
                        help="target image or directory, if directory analyse will run on all images within it,"
                             "default img")
    parser.add_argument("--visualize", type=bool, default=False,
                        help="if True object detection and age_gender estimation would be drawn on each image, "
                             "default False")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    start = time.time()
    image_path = get_args().image_path
    visualize = get_args().visualize
    result = run_object_detect(image_path, visualize=visualize)
    obj_detect_time = time.time()
    result += run_NSFW_detect(image_path)
    nsfw_time = time.time()
    result += run_scene_description(image_path)
    scene_desc_time = time.time()
    result += run_age_gender_detect(image_path, visualize=visualize)
    end = time.time()
    print(result)
    print(f"[INFO] Object detection took {obj_detect_time - start:.2f} seconds")
    print(f"[INFO] NSFW classification took {nsfw_time - obj_detect_time:.2f} seconds")
    print(f"[INFO] Scene classification took {scene_desc_time - nsfw_time:.2f} seconds")
    print(f"[INFO] Age-gender classification took {end - scene_desc_time:.2f} seconds")
    print(f"[INFO] Total classification took {end - start:.2f} seconds")



