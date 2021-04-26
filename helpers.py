"""Helpers functions for running inference on pre-trained models

This module provides functions that are useful for running inference
of the pre-trained neural nets.

This module contains following functions:
    * yield_images_from_dir - yields resized image/images from a path

"""

import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Tuple


def yield_images_from_dir(img_path: str, required_size: int = 640) \
        -> Generator[Tuple[np.ndarray, str], None, None]:
    """Post processes results of the NSFW classification and store them
    in a text file located at the results directory.

    Parameters
    ----------
    img_path : str
        Path to an image or dir. If dir, gradually yields all images in
        the dir.
    required_size : int, optional
        A new size of the yielded image. Yielded image resolution will be
        required_size * required_size. Default is 640.

    Yields
    ------
    np.ndarray
        Yielded image represented as numpy`s array
    str
        Name of the image with .format suffix

    """
    img_path = Path(img_path)
    search_pattern = '*.*'

    if img_path.is_file():
        search_pattern = img_path.name
        img_path = img_path.parent

    for file in img_path.glob(search_pattern):
        img = cv2.imread(str(file))

        if img is not None:
            yield cv2.resize(img, (required_size, required_size)), file.name
