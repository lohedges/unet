import tensorflow as tf

from model import unet
from data import trainGenerator, testGenerator, saveResult


data_gen_args = dict(
    rotation_range=15,  # degrees
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=5,  # degrees
    zoom_range=0.1,
    brightness_range=[0.9, 1.1],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='mirror',
    #preprocessing_function=foo,  # something to do the warping
    validation_split=0.1,  # Fraction of data to use for validation
)
train, validate = trainGenerator(2, 'data/filament/train', 'image', 'label', data_gen_args, save_to_dir=None)

model = unet()
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('unet_filament.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
model.fit(train, steps_per_epoch=100, epochs=30, callbacks=[model_checkpoint], validation_data=validate, validation_steps=10)

testGene = testGenerator("data/filament/test")
results = model.predict(testGene, verbose=1)
saveResult("data/filament/test", results)
