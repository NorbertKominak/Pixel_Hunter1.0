from pathlib import Path
import cv2
import dlib
import numpy as np
from omegaconf import OmegaConf
from age_gender_estimation.src.factory import get_model


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
            r = 640 / max(w, h)
            yield cv2.resize(img, (int(w * r), int(h * r))), file.name


def draw_rectangle(img, d, label):
    x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.rectangle(img, (x1, y1), (x1, y1), (255, 0, 0), thickness=2)
    cv2.putText(img, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, lineType=cv2.LINE_AA)


def run(img_path, visualize=False):
    weight_file = 'EfficientNetB3_224_weights.11-3.44.hdf5'

    # for face detection
    detector = dlib.get_frontal_face_detector()

    # load model and weights
    model_name, img_size = Path(weight_file).stem.split("_")[:2]
    img_size = int(img_size)
    cfg = OmegaConf.from_dotlist([f"model.model_name={model_name}", f"model.img_size={img_size}"])
    model = get_model(cfg)
    model.load_weights(weight_file)

    image_generator = yield_images_from_dir(img_path)
    result = ''

    for img, img_name in image_generator:
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(input_img)

        # detect faces using dlib detector
        detected = detector(input_img, 1)
        faces = np.empty((len(detected), img_size, img_size, 3))

        if len(detected) > 0:

            # predict ages and genders of the detected faces
            results = model.predict(faces)
            predicted_genders = results[0]
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()

            result += f'{img_name};\n'
            for i, d in enumerate(detected):
                label = f'{"M" if predicted_genders[i][0] < 0.5 else "F"} age {int(predicted_ages[i])}, '
                if visualize:
                    draw_rectangle(img, d, label)

                result += label
            result += '\n'

            if visualize:
                cv2.imshow('Image', img)
                cv2.waitKey()
                cv2.destroyAllWindows()

    return result
