"""Amazon Rekognition Module

This module provides functionality to use object detection, content
moderation, label detection and face detection models of the Amazon
Rekognition API.

This module contains following functions:

    * run_aws_api          - runs specified task on selected imgs
    * check_img            - checks img for API`s restrictions
    * detect_objects       - returns detected objects in a single image
    * age_gender           - returns age and gender estimation of faces
                             in a single image
    * scene_classification - returns detected labels in a single image
    * moderate_image       - returns image moderation labels of a
                             single image

"""

import boto3
import os
import cv2

# Image Restrictions
ALLOWED_FORMATS = {".jpg", ".jpeg", ".png"}
IMG_SIZE_LIMIT = 5e+6
MIN_IMG_DIM = 80


def run_aws_api(imgs, input_dir, output_dir, task_name):
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
        print("[AMAZON REKOGNITION] Connecting to a client...")
        client = boto3.client("rekognition")

    tasks = {"detect_objects": detect_objects,
             "age_gender": age_gender,
             "scene_classification": scene_classification,
             "moderate_image": moderate_image}
    task = tasks[task_name]

    output_name = f"aws_{task_name}.csv"
    with open(os.path.join(output_dir, output_name), "w") as output_file:
        if task_name == "detect_objects":
            output_file.write("img_name,label_name,score,left,top,right,"
                              "bottom\n")
        if task_name == "age_gender":
            output_file.write("img_name,age,gender,left,top,right,bottom\n")

        for img in imgs:
            check_result = check_img(img, input_dir)
            if check_result != "correct":
                output = check_result
            else:
                output = task(client, img, input_dir)

            output_file.write(output)

    print(f"[AMAZON REKOGNITION] Analyzed {len(imgs)} images. "
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
        return f"{img},[Error] Unsupported format\n"

    if os.path.getsize(os.path.join(input_dir, img)) >= IMG_SIZE_LIMIT:
        return f"{img},[Error] Size is larger than {IMG_SIZE_LIMIT}B\n"

    img_cv2 = cv2.imread(os.path.join(input_dir, img))
    img_height, img_width, _ = img_cv2.shape
    if img_height < MIN_IMG_DIM or img_width < MIN_IMG_DIM:
        return f"{img},[Error] Dim is smaller than {MIN_IMG_DIM}\n"

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
        response = client.detect_labels(Image={"Bytes": img_file.read()})

        img_cv2 = cv2.imread(os.path.join(input_dir, img))
        img_height, img_width, _ = img_cv2.shape

        if len(response["Labels"]) == 0:
            output = f"{img}\n"

        for label in response["Labels"]:
            if len(label["Instances"]) != 0:
                for instance in label["Instances"]:
                    left = int(instance['BoundingBox']['Left'] * img_width)
                    top = int(instance['BoundingBox']['Top'] * img_height)
                    right = left + int(instance['BoundingBox']['Width'] *
                                       img_width)
                    bottom = top + int(instance['BoundingBox']['Height'] *
                                       img_height)
                    output += f"{img},{label['Name']}," \
                              f"{label['Confidence']:.2f}," \
                              f"{left},{top},{right},{bottom}\n"

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
        response = client.detect_faces(Image={'Bytes': img_file.read()},
                                       Attributes=['ALL'])

        img_cv2 = cv2.imread(os.path.join(input_dir, img))
        img_height, img_width, _ = img_cv2.shape

        if len(response["FaceDetails"]) == 0:
            output = f"{img}\n"

        for faceDetail in response["FaceDetails"]:
            left = faceDetail['BoundingBox']['Left'] * img_width
            top = faceDetail['BoundingBox']['Top'] * img_height
            right = left + faceDetail['BoundingBox']['Width'] * img_width
            bottom = top + faceDetail['BoundingBox']['Height'] * img_height
            output += f"{img},{faceDetail['AgeRange']['Low']}-" \
                      f"{faceDetail['AgeRange']['High']}," \
                      f"{faceDetail['Gender']['Value'][0]}," \
                      f"{left:.0f},{top:.0f},{right:.0f},{bottom:.0f}\n"

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
        response = client.detect_labels(Image={'Bytes': img_file.read()})

        output = f"{img}"
        for label in response['Labels']:
            output += f",{label['Name']}:{label['Confidence']:.2f}"

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
        <img_name>[,<label>:<score>]\n

    """
    with open(os.path.join(input_dir, img), "rb") as img_file:
        response = client.detect_moderation_labels(
            Image={'Bytes': img_file.read()})

        output = f"{img}"
        for label in response['ModerationLabels']:
            output += f",{label['Name']}:{label['Confidence']:.2f}"

        output += "\n"

    return output
