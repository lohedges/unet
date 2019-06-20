from pathlib import Path

import mrcfile
import numpy as np
import skimage.io
import skimage.transform
import tensorflow as tf

my_model = tf.keras.models.load_model('my_model.hdf5')

filename = "/home/matt/proof_example_data/raw/FoilHole_24671230_Data_24671848_24671849_20181025_2350-80431.mrc"
file = Path(filename)

with mrcfile.open(file, permissive=True) as mrc:
    h = mrc.header
    d = mrc.data

d = skimage.transform.resize(d, (256, 256))

#d_cv = image.adjust_gradient(d_cv)

#d_cv = image.normalise_float(d_cv)
d_min = d.min()
d_max = d.max()
shifted = d - d_min
shifted /= d_max - d_min
d = shifted

d = d[np.newaxis, :, :, np.newaxis]  # Reshape to (batch size, x, y, layers)

results = my_model.predict(d, verbose=1)

skimage.io.imsave("blah.png", results[0, :, :, 0])
