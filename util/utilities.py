import os
import numpy as np
from matplotlib import image


def load_images_to_dict(path='./images/original/', suffix='.png') -> dict:
    '''
    Loads all png files found in path
    :param path: path to folder containing images
    :return: list of numpy arrays containing loaded png files
    '''

    assert os.path.exists(path)
    png_names = [f for f in os.listdir(path) if f.endswith(suffix)]
    imgs = {}

    if len(png_names) == 0:
        print(f"No PNG files found in {path}\n returning empty dict.")

    try:
        # try to load all images in png_names
        imgs = {f: image.imread(os.path.join(path, f)) for f in png_names}
    except FileNotFoundError as fnf:
        print(fnf)

    return imgs


def move_color_axis(img, source, target):

    return np.moveaxis(img, source, target)