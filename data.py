import os
import pathlib
import itertools
from functools import partial

import numpy as np
import skimage.io
import skimage.transform
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
import tensorflow as tf


def load_data(input_dir, label_dir, aug_dict, target_size=(256, 256), batch_size=2):
    input_images = pathlib.Path(input_dir).glob("*")
    label_images = pathlib.Path(label_dir).glob("*")
    input = DataFrame({"image": Series(input_images).apply(str), "mask": Series(label_images).apply(str)})

    train_filenames, test_filenames = train_test_split(input, test_size=0.1, random_state=1)

    train_filenames = train_filenames[:10]
    test_filenames = test_filenames[:2]

    def images(ds):
        out = []
        progbar = tf.keras.utils.Progbar(len(ds), unit_name="image")
        for filename in ds["image"]:
            out.append(load_resize_reshape(filename, target_size=target_size))
            progbar.add(1)
        return np.array(out)

    def masks(ds):
        out = []
        progbar = tf.keras.utils.Progbar(len(ds), unit_name="image")
        for filename in ds["mask"]:
            mask = load_resize_reshape(filename, target_size=target_size)
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0
            out.append(mask)
            progbar.add(1)
        return np.array(out)

    train_images = images(train_filenames)
    train_masks = masks(train_filenames)
    train, validation = train_generator(train_images, train_masks, batch_size, aug_dict, target_size=target_size)

    test_images = images(test_filenames)
    test_masks = masks(test_filenames)
    test = (test_images, test_masks)

    print(f"   Train examples: {len(train.x)}")
    print(f"Validate examples: {len(validation.x)}")
    print(f"    Test examples: {len(test_filenames)}")

    return train, validation, test


def train_generator(train_images, train_masks, batch_size, aug_dict, image_color_mode="grayscale", mask_color_mode="grayscale",
                    image_save_prefix="image", mask_save_prefix="mask", save_to_dir=None, target_size=(256, 256),
                    seed=1):
    image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**aug_dict)
    training = image_datagen.flow(
        train_images,
        train_masks,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed,
        subset="training",
    )
    validation = image_datagen.flow(
        train_images,
        train_masks,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed,
        subset="validation",
    )
    return training, validation


def load_resize_reshape(filename, target_size=(256, 256)):
    img = skimage.io.imread(filename, as_gray=True)
    img = skimage.transform.resize(img, target_size)
    img = img[:, :, np.newaxis]
    d_min = img.min()
    d_max = img.max()
    img -= d_min
    img /= d_max - d_min
    return img


def save_result(save_path, npyfile):
    for i, item in enumerate(npyfile):
        img = item[:, :, 0]
        skimage.io.imsave(os.path.join(save_path, "%d_predict.png" % i), img)
