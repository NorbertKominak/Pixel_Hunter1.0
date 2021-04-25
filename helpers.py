import cv2
from pathlib import Path


def yield_images_from_dir(img_path, required_size=640):
    img_path = Path(img_path)
    search_pattern = '*.*'

    if img_path.is_file():
        search_pattern = img_path.name
        img_path = img_path.parent

    for file in img_path.glob(search_pattern):
        img = cv2.imread(str(file))

        if img is not None:
            yield cv2.resize(img, (required_size, required_size)), file.name
