from pathlib import Path

import mrcfile
import numpy as np
import sklearn.linear_model
import skimage.io
import skimage.transform
import tensorflow as tf


def adjust_gradient(image):
    m, n = image.shape
    R, C = np.mgrid[:m, :n]
    out = np.column_stack((C.ravel(), R.ravel(), image.ravel()))

    reg = sklearn.linear_model.LinearRegression().fit(out[:, 0:2], out[:, 2])

    plane = np.fromfunction(lambda x, y: y * reg.coef_[0] + x * reg.coef_[1] + reg.intercept_, image.shape)

    return image - plane


my_model = tf.keras.models.load_model('my_model.hdf5')

filename = "/home/matt/proof_example_data/raw/FoilHole_24671230_Data_24671848_24671849_20181025_2350-80431.mrc"
file = Path(filename)

with mrcfile.open(file, permissive=True) as mrc:
    h = mrc.header
    d = mrc.data

d = skimage.transform.resize(d, (256, 256))

d = adjust_gradient(d)

#d_cv = image.normalise_float(d_cv)
d_min = d.min()
d_max = d.max()
shifted = d - d_min
shifted /= d_max - d_min
d = shifted

d = d[np.newaxis, :, :, np.newaxis]  # Reshape to (batch size, x, y, layers)

results = my_model.predict(d, verbose=1)

skimage.io.imsave("blah.png", results[0, :, :, 0])
