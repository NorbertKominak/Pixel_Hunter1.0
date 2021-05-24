"""Age and Gender Estimation Module

This module provides functionality to run inference of the Age and
Gender Estimation model using the run() function. If allowed it also
sends selected images to face detection models of the Amazon
Rekognition, Google Vision Cloud and Microsoft Computer Vision APIs.

This module contains following functions:

    * get_model - returns a model object with inference features
    * run - runs inference on images and stores results

"""

import cv2
import dlib
import os
import numpy as np
from tensorflow.keras import applications
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from pathlib import Path
from awsvision import run_aws_api
from msvision import run_ms_api

# Path to the pre-trained model file saved in .hdf5 format
MODEL_PATH = "models/EfficientNetB3_224_weights.11-3.44.hdf5"

# Specifies relative margin around detected faces for age-gender estimation
MARGIN = 0.4

# If predicted age is lower than this threshold, the image will be sent
# for further analysis to APIs
MINIMUN_AGE_THRESHOLD = 25

# If predicted gender is male and predicted age is above this threshold
# the image will be sent for further analysis to APIs
MAX_MALE_AGE_THRESHOLD = 50


def get_model(model_name, input_size):
    """Instantiates a base pre-trained model, specifies its input,
    output layers and returns a model object with training and
    inference features.

    Parameters
    ----------
    model_name : str
        Model name
    input_size : int
        Required image input size of the model

    Returns
    -------
        A model object with training and
        inference features.

    """
    # Instantiates the {model_name} architecture, specifying input
    # tuple (224, 224, 3) and adds avg pooling at
    # the output of the network
    base_model = getattr(applications, model_name)(
        include_top=False,
        input_shape=(input_size, input_size, 3),
        pooling="avg")

    features = base_model.output

    # Adds a new dense layer as the output layer for gender_estimation,
    # 2 neurons Male/Female
    pred_gender = Dense(units=2, activation="softmax",
                        name="pred_gender")(features)

    # Adds a new dense layer as the new output layer for
    # gender_estimation, 101 neurons each representing specific age
    pred_age = Dense(units=101, activation="softmax",
                     name="pred_age")(features)

    # Groups layers into an object with training and inference features
    model = Model(inputs=base_model.input, outputs=[pred_gender, pred_age])
    return model


def run(input_dir, output_dir, allow_api):
    """ Runs inference of the Age and Gender Estimation model on images
    in the input_dir. Stores results in a .csv file located in the
    output_dir directory. If allowed, run() sends selected images to
    face detection models of the Amazon Rekognition, Google Vision
    Cloud and Microsoft Computer Vision APIs. Only images with any
    predicted age below the MINIMUN_AGE_THRESHOLD constant or with any
    predicted gender=male and age above the MAX_MALE_AGE_THRESHOLD
    constant are sent to APIs. APIs` results are also stored in
    the output_dir.

    Parameters
    ----------
    input_dir : str
        Path to the dir with images to run inference on.
    output_dir : str
        Path to the dir where all results will be stored.
    allow_api : bool
        If True selected images are sent to the APIs.

    """
    # Return dlib`s default frontal face detector
    detector = dlib.get_frontal_face_detector()

    model_name, input_size = Path(MODEL_PATH).stem.split("_")[:2]
    input_size = int(input_size)
    print("[AGE AND GENDER] Loading model...")
    model = get_model(model_name, input_size)
    model.load_weights(MODEL_PATH)
    imgs_for_api = set()
    count_img = 0

    with open(os.path.join(output_dir, "age_gender.csv"), "w") as output_file:
        output_file.write("img_name,age,gender,left,top,right,bottom\n")
        for file in os.listdir(input_dir):
            img = cv2.imread(os.path.join(input_dir, file))
            if img is not None:
                count_img += 1
                # BGR is default OpenCV color space
                input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_height, img_width, _ = np.shape(input_img)

                # Detect faces using dlib`s default frontal face detector
                detected = detector(input_img, 1)
                faces = np.empty((len(detected), input_size, input_size, 3))

                if len(detected) > 0:
                    # Each detected face is copied and resized into np.array
                    for i, box in enumerate(detected):
                        x1, y1, x2, y2, w, h = (box.left(), box.top(),
                                                box.right() + 1,
                                                box.bottom() + 1,
                                                box.width(), box.height())
                        xw1 = max(int(x1 - MARGIN * w), 0)
                        yw1 = max(int(y1 - MARGIN * h), 0)
                        xw2 = min(int(x2 + MARGIN * w), img_width - 1)
                        yw2 = min(int(y2 + MARGIN * h), img_height - 1)
                        faces[i] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1],
                                              (input_size, input_size))

                    predictions = model.predict(faces)
                    predicted_genders = predictions[0]
                    ages = np.arange(0, 101).reshape(101, 1)
                    predicted_ages = predictions[1].dot(ages).flatten()

                    for i, box in enumerate(detected):
                        age = int(predicted_ages[i])
                        gender = "M" if predicted_genders[i][0] < 0.5 else "F"
                        output = f"{file},{age},{gender},{box.left()}," \
                                 f"{box.top()},{box.right()},{box.bottom()}\n"

                        if ((allow_api and age < MINIMUN_AGE_THRESHOLD) or
                                (age > MAX_MALE_AGE_THRESHOLD and
                                 gender == "M")):
                            imgs_for_api.add(file)

                        output_file.write(output)
                else:
                    output_file.write(f"{file}\n")

    print(f"[AGE AND GENDER] Analyzed {count_img} images, results are "
          f"stored in {os.path.join(output_dir, 'age_gender.csv')}")
    print(f"[AGE AND GENDER] Passing {len(imgs_for_api)} images for further"
          " analysis to APIs...")

    run_aws_api(imgs_for_api, input_dir, output_dir, "age_gender")
    run_ms_api(imgs_for_api, input_dir, output_dir, "age_gender")
    print("-------------------------------------------------------\n")
