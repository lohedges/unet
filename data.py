import numpy as np
import os

import tensorflow as tf
import skimage.io as io
import skimage.transform as trans


def adjust_data(img, mask):
    img = img / 255
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return img, mask


def train_generator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode="grayscale",
                    mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                    save_to_dir=None, target_size=(256, 256), seed=1):
    """
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    """
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(**aug_dict)
    kwargs = {
        "directory": train_path,
        "class_mode": None,
        "target_size": target_size,
        "batch_size": batch_size,
        "save_to_dir": save_to_dir,
        "seed": seed,
    }
    image_generator = datagen.flow_from_directory(
        classes=[image_folder],
        color_mode=image_color_mode,
        save_prefix=image_save_prefix,
        subset="training",
        **kwargs,
    )
    mask_generator = datagen.flow_from_directory(
        classes=[mask_folder],
        color_mode=mask_color_mode,
        save_prefix=mask_save_prefix,
        subset="training",
        **kwargs,
    )
    val_image_generator = datagen.flow_from_directory(
        classes=[image_folder],
        color_mode=image_color_mode,
        save_prefix=image_save_prefix,
        subset="validation",
        **kwargs,
    )
    val_mask_generator = datagen.flow_from_directory(
        classes=[mask_folder],
        color_mode=mask_color_mode,
        save_prefix=mask_save_prefix,
        subset="validation",
        **kwargs,
    )
    train_gen = (adjust_data(img, mask) for img, mask in zip(image_generator, mask_generator))
    valid_gen = (adjust_data(img, mask) for img, mask in zip(val_image_generator, val_mask_generator))
    return train_gen, valid_gen


def test_generator(test_path, target_size=(256, 256), as_gray=True):
    filenames = ["FoilHole_24681291_Data_24671848_24671849_20181025_0148-78831.png"]
    for filename in filenames:
        img = io.imread(os.path.join(test_path, filename), as_gray=as_gray)
        img = img / 255
        img = trans.resize(img, target_size)
        img = np.reshape(img, img.shape+(1,))
        img = np.reshape(img, (1,)+img.shape)
        yield img


def save_result(save_path, npyfile):
    for i, item in enumerate(npyfile):
        img = item[:, :, 0]
        io.imsave(os.path.join(save_path, "%d_predict.png" % i), img)
