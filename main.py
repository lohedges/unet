import os
from datetime import datetime
from pathlib import Path

import tensorflow as tf

from model import unet
from data import train_generator, test_generator, save_result

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

job_id = os.environ.get("PBS_JOBID", datetime.now().isoformat(timespec="seconds"))

cwd = Path.cwd()
output_dir = cwd / job_id
output_dir.mkdir()

log_dir = cwd/"logs"/job_id
log_dir.mkdir(parents=True)

print(f"Writing output to {output_dir}")
print(f"Tensorboard logs written to {log_dir}")

batch_size = 2
images_per_epoch = 200
steps_per_epoch = images_per_epoch // batch_size
epochs = 1000
learning_rate = 1e-4

print(f"      Batch size: {batch_size}")
print(f"Images per epoch: {images_per_epoch}")
print(f" Steps per epoch: {steps_per_epoch}")
print(f"          Epochs: {epochs}")

print(f"   Learning rate: {learning_rate}")

data_gen_args = dict(
    rotation_range=45,  # degrees
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=15,  # degrees
    zoom_range=0.1,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='mirror',
    #preprocessing_function=foo,  # something to do the warping
    validation_split=0.1,  # Fraction of data to use for validation
)
train, validate = train_generator(batch_size, 'data/filament/train', 'image', 'label', data_gen_args, save_to_dir=None)

model = unet(learning_rate=learning_rate)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(str(output_dir/'unet_filament.hdf5'), monitor='val_loss', verbose=1, save_best_only=True)
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=10)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_images=True)
model.fit(train, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=[model_checkpoint, es, tensorboard], validation_data=validate, validation_steps=1)

testGene = test_generator("data/filament/test")
results = model.predict(testGene, verbose=1)
save_result(output_dir, results)
