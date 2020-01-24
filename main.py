import numpy as np
import skimage.io
import tensorflow as tf

import data
import model

image_size = (256, 256)
batch_size = 2

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        #for gpu in gpus:
        #    tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


data_gen_args = dict(
    rotation_range=15,  # degrees?
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=5,  # degrees?
    zoom_range=0.1,
    brightness_range=[0.9, 1.1],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='mirror',  # reflect?
    #preprocessing_function=foo,  # something to do the warping
    validation_split=0.1,  # Fraction of data to use for validation
)
train, validate, test = data.load_data(
    "/home/matt/proof_example_data/unet_foo/input_png",
    "/home/matt/proof_example_data/unet_foo/masks",
    data_gen_args,
    target_size=image_size,
    batch_size=batch_size,
)

print(f"Training batch size: {train.batch_size}")

my_model = model.unet(input_size=image_size + (1,))
#my_model.load_weights('unet_filaments.hdf5')
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('unet_filaments.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)

print("# Fitting")

steps_per_epoch = len(train.x) // train.batch_size

print(f"Steps per epoch: {steps_per_epoch}")

my_model.fit(
    train,
    steps_per_epoch=steps_per_epoch,
    epochs=2,  # 5
    callbacks=[model_checkpoint, tensorboard],
    validation_data=validate,
)

print("# Testing")

test_loss, test_acc, test_iou = my_model.evaluate(*test, verbose=0)
print(f"    Loss: {test_loss:4.3f}\nAccuracy: {test_acc*100:4.1f}%\n     IoU: {test_iou*100:4.1f}%")

print("# Saving")

my_model.save('my_model.hdf5')

print("# Predicting")

example_data = data.load_resize_reshape("/home/matt/proof_example_data/unet_foo/input_png/FoilHole_24677417_Data_24671816_24671817_20181025_1806-79997.png")
example_data = example_data[np.newaxis, :, :, :]
results = my_model.predict(example_data, verbose=1)
skimage.io.imsave("/home/matt/proof_example_data/unet_foo/test_output/blah.png", results[0, :, :, 0])
