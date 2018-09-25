import os
import glob
import numpy as np
from PIL import Image


def load_imagenet_train():
    """Warning: generated array is about 16GB."""
    np_filename = 'data/imagenet/train_64x64.npy'
    if os.path.exists(np_filename):
        return np.load(np_filename)
    else:
        filenames = glob.glob('data/imagenet/train_64x64/*')
        n = len(filenames)
        first_img = np.array(Image.open(filenames[0]))
        h, w, z = first_img.shape
        dtype = first_img.dtype
        data = np.empty((n, h, w, z), dtype)
        for i, name in enumerate(filenames):
            if i % 10000 == 0:
                print("Decompressed {} images.".format(i))
            data[i] = np.array(Image.open(name))
        np.save(np_filename, data)
        return data

def load_imagenet_valid():
    np_filename = 'data/imagenet/valid_64x64.npy'
    if os.path.exists(np_filename):
        return np.load(np_filename)
    else:
        filenames = glob.glob('data/imagenet/valid_64x64/*')
        n = len(filenames)
        first_img = np.array(Image.open(filenames[0]))
        h, w, z = first_img.shape
        dtype = first_img.dtype
        data = np.empty((n, h, w, z), dtype)
        for i, name in enumerate(filenames):
            if i % 10000 == 0:
                print("Decompressed {} images.".format(i))
            data[i] = np.array(Image.open(name))
        np.save(np_filename, data)
        return data
