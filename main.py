import skimage.io
import tensorflow as tf

import data
import model

data_gen_args = dict(
    rotation_range=15,  # degrees?
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=5,  # degrees?
    zoom_range=0.1,
    brightness_range=[0.9, 1.1],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='reflect',  # reflect?
    #preprocessing_function=foo,  # something to do the warping
)
# data_gen_args = dict(
#     rotation_range=0.2,
#     width_shift_range=0.05,
#     height_shift_range=0.05,
#     shear_range=0.05,
#     zoom_range=0.05,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )
train, validate, test = data.load_data("/home/matt/proof_example_data/unet_foo/input_png", "/home/matt/proof_example_data/unet_foo/masks", data_gen_args)

my_model = model.unet()
#my_model.load_weights('unet_filaments.hdf5')
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('unet_filaments.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_images=True, write_graph=True)

print("# Fitting")

my_model.fit_generator(
    train,
    steps_per_epoch=500,
    epochs=5,
    callbacks=[model_checkpoint, tensorboard],
    validation_data=validate,
)

print("# Testing")

test_loss, test_acc = my_model.evaluate(test, verbose=1)
print(f"\n\nAccuracy: {test_acc*100:4.1f}%\n")

print("# Saving")

my_model.save('my_model.hdf5')

print("# Predicting")

results = my_model.predict(data.load_resize_reshape("/home/matt/proof_example_data/unet_foo/input_png/FoilHole_24677417_Data_24671816_24671817_20181025_1806-79997.png"), verbose=1)
skimage.io.imsave("/home/matt/proof_example_data/unet_foo/test_output/blah.png", results[0, :, :, 0])
