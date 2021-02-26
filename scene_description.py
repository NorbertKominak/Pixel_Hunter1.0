import cv2
import numpy as np
from pathlib import Path


def load_classes():
    # Scene classification load classes
    classes = []
    outdoors_indoors = []
    with open('IO_places365.txt') as labels_file:
        for line in labels_file:
            classes.append(line[3:line.find(' ')])
            outdoors_indoors.append(line[line.find(' '):].strip())

    return classes, outdoors_indoors


def yield_images_from_dir(img_path):
    img_path = Path(img_path)
    search_pattern = '*.*'
    if img_path.is_file():
        search_pattern = img_path.name
        img_path = img_path.parent

    for file in img_path.glob(search_pattern):
        img = cv2.imread(str(file))
        if img is not None:
            yield img, file.name


def run(img_path):
    classes, outdoors_indoors = load_classes()
    scene_classification_net = cv2.dnn.readNetFromCaffe('deploy_resnet152_places365.prototxt',
                                                        'resnet152_places365.caffemodel')

    image_generator = yield_images_from_dir(img_path)
    result = ''
    for img, img_name in image_generator:
        scene_classification_net.setInput(cv2.dnn.blobFromImage(img, 1, (224, 224), (104, 117, 123)))
        network_output = scene_classification_net.forward()

        # stores indices of top-5 scene classification classes
        idxs = np.argsort(network_output[0])[::-1][:5]
        result += f'{img_name};\n'
        for (i, idx) in enumerate(idxs):
            result += f"Label with probability rank {i + 1}: {classes[idx]} {network_output[0][idx] * 100:.2f}%, "
            result += f"{'indoors' if outdoors_indoors[idx] == 1 else 'outdoors'}\n"

    return result
