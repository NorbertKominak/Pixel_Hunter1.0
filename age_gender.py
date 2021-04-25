"""Age and Gender Estimation

This module provides functionality to run inference of the
Age and Gender Estimation model using the run() function.

This module contains following functions:

    * get_model - returns a model object with inference features
    * draw_rectangle - draws a rectangle with a label around
                       a detected object
    * run - runs inference on an image/images and stores results

"""

import cv2
import dlib
import numpy as np
import helpers
from tensorflow.keras import applications
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from os import environ
from pathlib import Path

# Path to the pre-trained model file saved in .hdf5 format
MODEL_PATH = "models/EfficientNetB3_224_weights.11-3.44.hdf5"

# Specifies relative margin around detected faces for age-gender estimation
MARGIN = 0.4

# To silence TensorFlow warnings
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_model(model_name: str, input_size: int) -> Model:
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
    Model
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


def draw_rectangle(img: np.ndarray, box: dlib.rectangles, label: str)\
        -> None:
    """Draws a rectangle with a label around detected object

    Parameters
    ----------
    img : np.ndarray
        Image representation as an numpy array.
    box : dlib.rectangles
        Dlib detector`s result with coordinates of the rectangle.
    label : str
        Label of the detected object.

    """
    x1 = box.left()
    y1 = box.top()
    x2 = box.right() + 1
    y2 = box.bottom() + 1
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(img, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, lineType=cv2.LINE_AA)


def run(img_path: str, visualize=False) -> None:
    """ Runs inference of the Age and Gender Estimation model on images
    at img_path. Stores results in a text file located in the results
    directory. If visualize is True, stores visualized results in the
    results/visualize/age_gender directory.

    Parameters
    ----------
    img_path : str
        Path to an image or dir. If dir, then runs inference on all
        images located within the dir.
    visualize : bool, optional
        Flag whether visualized result should be stored as well.
        Default False.

    """
    # Return dlib`s default frontal face detector
    detector = dlib.get_frontal_face_detector()

    # tf.keras.backend.clear_session()

    model_name, input_size = Path(MODEL_PATH).stem.split("_")[:2]
    input_size = int(input_size)
    model = get_model(model_name, input_size)
    model.load_weights(MODEL_PATH)

    image_generator = helpers.yield_images_from_dir(img_path)
    with open("results/age_gender.txt", "w") as output_file:
        for img, img_name in image_generator:
            input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = np.shape(input_img)

            # Detect faces using dlib`s default frontal face detector
            detected = detector(input_img, 1)
            faces = np.empty((len(detected), input_size, input_size, 3))

            if len(detected) > 0:
                # Each detected face is copied and resized into faces np.array
                for i, box in enumerate(detected):
                    x1, y1, x2, y2, w, h = box.left(), box.top(), \
                                           box.right() + 1, box.bottom() + 1, \
                                           box.width(), box.height()
                    xw1 = max(int(x1 - MARGIN * w), 0)
                    yw1 = max(int(y1 - MARGIN * h), 0)
                    xw2 = min(int(x2 + MARGIN * w), img_w - 1)
                    yw2 = min(int(y2 + MARGIN * h), img_h - 1)
                    faces[i] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1],
                                          (input_size, input_size))

                predictions = model.predict(faces)
                predicted_genders = predictions[0]
                ages = np.arange(0, 101).reshape(101, 1)
                predicted_ages = predictions[1].dot(ages).flatten()

                result = f'{img_name};\n'
                for i, box in enumerate(detected):
                    label = f'{"M" if predicted_genders[i][0] < 0.5 else "F"}'\
                            f', {int(predicted_ages[i])}, '

                    if visualize:
                        draw_rectangle(img, box, label)
                        cv2.imwrite(f'results/visualize/age_gender/'
                                    f'{img_name}', img)

                    result += label
                result += '\n'

                output_file.write(result)
