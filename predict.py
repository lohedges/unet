import tensorflow as tf

import data

my_model = tf.keras.models.load_model('my_model.hdf5')

filename = "/home/matt/proof_example_data/unet_foo/input_png/FoilHole_24671230_Data_24671848_24671849_20181025_2350-80431.png"

results = my_model.predict(data.load_resize_reshape(filename), verbose=1)
data.save_result("/home/matt/proof_example_data/unet_foo/test_output", results)
