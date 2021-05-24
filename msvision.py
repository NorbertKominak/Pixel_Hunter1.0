"""Microsoft Computer Vision Module

This module provides functionality to use object detection, content
moderation, label detection and face detection models of the Microsoft
Computer Vision API.

This module contains following functions:

    * run_ms_api          - runs specified task on selected imgs
    * check_img            - checks img for API`s restrictions
    * detect_objects       - returns detected objects in a single image
    * age_gender           - returns age and gender estimation of faces
                             in a single image
    * scene_classification - returns detected labels in a single image
    * moderate_image       - returns image moderation labels of a
                             single image

"""
import os
import cv2
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials

# Image Restrictions
ALLOWED_FORMATS = (".jpg", ".jpeg", ".png", ".gif", ".bnp")
IMG_SIZE_LIMIT = 4e+6
MAX_IMG_DIM = 1e+4
MIN_IMG_DIM = 50

# Sets identification credentials required for the communication
# with API
SUBSCRIPTION_KEY = "add_your_key"
ENDPOINT = "add_your_endpoint"


def run_ms_api(imgs, input_dir, output_dir, task_name):
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
        print("[MS COMPUTER VISION] Connecting to a client...")
        credentials = CognitiveServicesCredentials(SUBSCRIPTION_KEY)
        client = ComputerVisionClient(ENDPOINT, credentials)

    tasks = {"detect_objects": detect_objects,
             "age_gender": age_gender,
             "scene_classification": scene_classification,
             "moderate_image": moderate_image}
    task = tasks[task_name]

    output_name = f"ms_{task_name}.csv"
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

    print(f"[MS COMPUTER VISION] Analyzed {len(imgs)} images. "
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
    if (not MAX_IMG_DIM > img_height > MIN_IMG_DIM or
            not MAX_IMG_DIM > img_width > MIN_IMG_DIM):
        return f"{img},[Error] Img dim must be in between " \
               f"{MIN_IMG_DIM}-{MAX_IMG_DIM}\n"

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
        detect_results = client.detect_objects_in_stream(img_file)

        if len(detect_results.objects) == 0:
            output = f"{img}\n"

        for det_object in detect_results.objects:
            output += f"{img},{det_object.object_property}," \
                      f"{det_object.confidence:.2f}," \
                      f"{det_object.rectangle.x},{det_object.rectangle.y}," \
                      f"{det_object.rectangle.x + det_object.rectangle.w}," \
                      f"{det_object.rectangle.y + det_object.rectangle.h}\n"

    return output


def age_gender(client, img, input_dir):
    """ Runs face detection on the img.

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
        Detected faces in the img in the format:
        <img_name>,<age>,<gender>,<left>,<top>,<right>,<bottom>\n
        If none faces were detected only "<img_name>\n" is returned.

    """
    output = ""

    with open(os.path.join(input_dir, img), "rb") as img_file:
        detect_results = client.analyze_image_in_stream(img_file, ["faces"])

        if len(detect_results.faces) == 0:
            output = f"{img}\n"

        for face in detect_results.faces:
            left = face.face_rectangle.left
            top = face.face_rectangle.top
            right = left + face.face_rectangle.width
            bottom = top + face.face_rectangle.height
            output += f"{img},{face.age},{face.gender[0]}" \
                      f",{left},{top},{right},{bottom}\n"

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
        tags_result = client.tag_image_in_stream(img_file)

        output = f"{img}"
        for tag in tags_result.tags:
            output += f",{tag.name}:{tag.confidence:.2f}"

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
        <img_name><adult:score><racy:score>\n

    """
    with open(os.path.join(input_dir, img), "rb") as img_file:
        moderate_result = client.analyze_image_in_stream(img_file, ["adult"])

        output = f"{img},adult:{moderate_result.adult.adult_score:.2f}," \
                 f"racy:{moderate_result.adult.racy_score:.2f}\n"

    return output
