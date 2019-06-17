from tensorflow.keras.callbacks import ModelCheckpoint

import model
import data

data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     fill_mode='nearest')
myGene = data.trainGenerator(1, '/home/matt/proof_example_data/unet_foo', 'input_png', 'masks', data_gen_args, save_to_dir="/home/matt/proof_example_data/unet_foo/augmented/")

my_model = model.unet()
model_checkpoint = ModelCheckpoint('unet_filaments.hdf5', monitor='loss', verbose=1, save_best_only=True)
my_model.fit_generator(myGene, steps_per_epoch=300, epochs=1, callbacks=[model_checkpoint])

testGene = data.testGenerator("/home/matt/proof_example_data/unet_foo/input_png")
results = my_model.predict_generator(testGene, 30, verbose=1)
data.saveResult("/home/matt/proof_example_data/unet_foo/test_output", results)
