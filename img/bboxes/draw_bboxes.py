"""
This script converts csv results from the Object Detection model
and Age and Gender Estimation model into images with bounding boxes
and labels.

IT REWRITES IMAGES IN THE input_dir, USE COPIES!

Object Detection input file is in format
<img_name>,<label_name>,<score>,<left>,<top>,<right>,<bottom>
separate line for a single detected object.

Age and Gender Estimation input file is in format
<img_name>,<age>,<gender>,<left>,<top>,<right>,<bottom>
separate line for a single detected face.
"""

import os
import cv2


def draw_rectangle(img, x1, y1, x2, y2, label):
    """Draws a rectangle with a label around detected object
    """

    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(img, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, lineType=cv2.LINE_AA)


csv_object_det_input = "../outputs/det_object.csv"
csv_age_gender_input = "../outputs/age_gender.csv"
# Images in this dir will be rewritten by its copies with
# bounding boxes.
img_dir = "../.."

# Object Detection Bboxes
with open(csv_object_det_input) as csv_file:
    csv_file.readline()
    for line in csv_file:
        if "," not in line:
            continue

        img_name, label, score, x1, y1, x2, y2 = line.split(",")
        img = cv2.imread(os.path.join(img_dir, img_name))
        draw_rectangle(img, int(x1), int(y1), int(x2), int(y2),
                       f'{label} {float(score):.2f}%')
        cv2.imwrite(os.path.join(img_dir, img_name), img)

# Age and Gender Bboxes
with open(csv_age_gender_input) as csv_file:
    csv_file.readline()
    for line in csv_file:
        if "," not in line:
            continue

        img_name, age, gender, x1, y1, x2, y2 = line.split(",")
        img = cv2.imread(os.path.join(img_dir, img_name))
        draw_rectangle(img, int(x1), int(y1), int(x2), int(y2),
                       f'{gender}, {age}')
        cv2.imwrite(os.path.join(img_dir, img_name), img)
