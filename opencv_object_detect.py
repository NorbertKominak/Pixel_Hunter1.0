import cv2
from pathlib import Path


def load_labels(labels_path):
    labels = {}
    with open(labels_path) as labels_file:
        id = -1
        for line in labels_file:
            if line.find('id') != -1:
                id = int(line.split(':')[1].strip())
                labels[id] = ''
                continue

            if line.find('display_name') != -1:
                labels[id] = line.split(':')[1].strip().replace('"', '')

    return labels


def yield_images_from_dir(img_path):
    img_path = Path(img_path)
    search_pattern = '*.*'
    if img_path.is_file():
        search_pattern = img_path.name
        img_path = img_path.parent

    for file in img_path.glob(search_pattern):
        img = cv2.imread(str(file))
        if img is not None:
            h, w, _ = img.shape
            r = 512 / max(w, h)
            yield cv2.resize(img, (512, 512)), file.name


def draw_rectangle(img, points, label):
    rows, cols, channels = img.shape
    left = points[3] * cols
    top = points[4] * rows
    right = points[5] * cols
    bottom = points[6] * rows

    cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)
    cv2.putText(img, label, (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def run(img_path, visualize=False):
    labels = load_labels('labels/mscoco_label_map.pbtxt')
    object_detect_net = cv2.dnn.readNetFromTensorflow('faster_rcnn_frozen_inference_graph.pb',
                                                      'faster_rcnn_graph.pbtxt')

    image_generator = yield_images_from_dir(img_path)
    result = ''
    for img, img_name in image_generator:
        # Change to blobFromImages!
        object_detect_net.setInput(cv2.dnn.blobFromImage(img, size=(512, 512), mean=(104, 117, 123)))
        object_detect_output = object_detect_net.forward()

        result += f'{img_name};\n'
        n = 0
        for detection in object_detect_output[0, 0]:
            score = float(detection[2])
            if score > 0.3:
                label = f'  {score * 100:.0f}% {labels[int(detection[1] + 1)]}'
                n += 1
                if visualize:
                    draw_rectangle(img, detection, label)

                result += label

        result += f'Amount of objects detected: {n}\n'
        if visualize:
            cv2.imshow('Image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return result
