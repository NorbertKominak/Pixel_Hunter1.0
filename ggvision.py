"""Google CLoud Vision Module

This module provides functionality to use object detection, content
moderation, label detection and face detection models of the Google
Cloud Vision API.

This module contains following functions:

    * run_gg_api          - runs specified task on selected imgs
    * check_img            - checks img for API`s restrictions
    * detect_objects       - returns detected objects in a single image
    * scene_classification - returns detected labels in a single image
    * moderate_image       - returns image moderation labels of a
                             single image

"""

import os
import cv2
from google.cloud import vision

# Image Restrictions
ALLOWED_FORMATS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}
IMG_SIZE_LIMIT = 2e+7
MAX_PIXEL_LIMIT = 75e+6

# Sets identification credentials required for the communication
# with API
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "add_path"


def run_gg_api(imgs, input_dir, output_dir, task_name):
    """ Runs a single model specified by task_name on collection of
    images listed in imgs. Images are located in the input_dir. Outputs
    are stored as .csv file in the output_dir.

    Parameters
    ----------
    imgs : set[str]
        Set of images names for inference.
    input_dir : str
        Path to the dir with images to run inference on.
    output_dir : str
        Path to the dir where all results will be stored.
    task_name : str
        Name of the task. It specifies what model will be used for the
        inference.

    """
    if len(imgs) != 0:
        print("[GOOGLE CLOUD VISION] Connecting to a client...")
        client = vision.ImageAnnotatorClient()

    tasks = {"detect_objects": detect_objects,
             "scene_classification": scene_classification,
             "moderate_image": moderate_image}
    task = tasks[task_name]

    output_name = f"gg_{task_name}.csv"
    with open(os.path.join(output_dir, output_name), "w") as output_file:

        if task_name == "detect_objects":
            output_file.write("img_name,label_name,score,left,top,right"
                              ",bottom\n")
        if task_name == "age_gender":
            output_file.write("img_name,age,gender,left,top,right,bottom\n")

        for img in imgs:
            check_result = check_img(img, input_dir)
            if check_result != "correct":
                output = check_result
            else:
                output = task(client, img, input_dir)

            output_file.write(output)

    print(f"[GOOGLE CLOUD VISION] Analyzed {len(imgs)} images. "
          f"Results are stored in {os.path.join(output_dir, output_name)}")


def check_img(img, input_dir):
    """ Checks whether the img complies with API`s restrictions.

    Parameters
    ----------
    img : str
        Image name.
    input_dir : str
        Path to the dir with the image to check.

    Returns
    -------
        Error message if image does not comply with API`s
        restrictions. Otherwise, returns "correct".

    """
    img_format = img[img.find("."):].lower()
    if img_format not in ALLOWED_FORMATS:
        return f"{img},[Error] Unsupported format {img_format}\n"

    if os.path.getsize(os.path.join(input_dir, img)) >= IMG_SIZE_LIMIT:
        return f"{img},[Error] Size is larger than {IMG_SIZE_LIMIT}B\n"

    img_cv2 = cv2.imread(os.path.join(input_dir, img))
    img_height, img_width, _ = img_cv2.shape
    if img_height * img_width > MAX_PIXEL_LIMIT:
        return f"{img},[Error] Img has more than {MAX_PIXEL_LIMIT} pixels\n"

    return "correct"


def detect_objects(client, img, input_dir):
    """ Runs object detection on the img.

    Parameters
    ----------
    client : object
        Client for communication with API.
    img : str
        Image name.
    input_dir : str
        Path to the dir with the image.

    Returns
    -------
        Detected objects in the img in the format:
        <img_name>,<label_name>,<score>,<left>,<top>,<right>,<bottom>\n
        If none objects were detected only "<img_name>\n" is returned.

    """
    output = ""

    with open(os.path.join(input_dir, img), "rb") as img_file:
        content = img_file.read()
        image = vision.Image(content=content)
        objects = client.object_localization(
            image=image).localized_object_annotations

        img_cv2 = cv2.imread(os.path.join(input_dir, img))
        img_height, img_width, _ = img_cv2.shape

        if len(objects) == 0:
            output = f"{img}\n"

        for object_ in objects:
            left = int(object_.bounding_poly.normalized_vertices[0].x *
                       img_width)
            top = int(object_.bounding_poly.normalized_vertices[0].y *
                      img_height)
            right = int(object_.bounding_poly.normalized_vertices[2].x *
                        img_width)
            bottom = int(object_.bounding_poly.normalized_vertices[2].y *
                         img_height)
            output += f"{img},{object_.name},{object_.score:.2f},{left}," \
                      f"{top},{right},{bottom}\n"

    return output


def scene_classification(client, img, input_dir):
    """ Runs label detection on the img.

    Parameters
    ----------
    client : object
        Client for communication with API.
    img : str
        Image name.
    input_dir : str
        Path to the dir with the image.

    Returns
    -------
        Detected labels in the img in the format:
        <img_name>[,<label>:<score>]\n

    """
    with open(os.path.join(input_dir, img), "rb") as img_file:
        content = img_file.read()
        image = vision.Image(content=content)
        response = client.label_detection(image=image)
        labels = response.label_annotations

        output = f"{img}"
        for label in labels:
            output += f",{label.description}:{label.score:.2f}"

        output += "\n"

    return output


def moderate_image(client, img, input_dir):
    """ Runs image moderation on the img.

    Parameters
    ----------
    client : object
        Client for communication with API.
    img : str
        Image name.
    input_dir : str
        Path to the dir with the image.

    Returns
    -------
        Detected content moderation labels in the img in the format:
        <img_name>[,<label>:<likelihood_name>]\n

    """
    with open(os.path.join(input_dir, img), "rb") as img_file:
        content = img_file.read()
        image = vision.Image(content=content)
        response = client.safe_search_detection(image=image)
        labels = response.safe_search_annotation

        likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                           'LIKELY', 'VERY_LIKELY')

        output = f"{img},adult:{likelihood_name[labels.adult]}," \
                 f"medical:{likelihood_name[labels.medical]}," \
                 f"spoof:{likelihood_name[labels.spoof]}," \
                 f"violence:{likelihood_name[labels.violence]}," \
                 f"racy:{likelihood_name[labels.racy]}\n"

    return output
