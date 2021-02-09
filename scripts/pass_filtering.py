from os import makedirs
from os.path import join, pardir, exists

import numpy as np

from ot_sparse_projection import ot_sparse_projection, misc
from ot_sparse_projection.dictionaries import get_filter_handler

filter_type = 'dct'
gamma = .1
n = 256
imName = 'camera'
folder = join(pardir, 'img', 'filtering')
if not exists(folder):
    makedirs(folder)

k = int(n / 4)

im, scaling = misc.get_image(imName, n)
filter_handler = get_filter_handler(im, filter_type)

Y, Z, obj = ot_sparse_projection.wasserstein_image_low_pass(im, filter_handler, k, gamma)

print(Y.sum())
ZZ = Z.copy()
recons = filter_handler.dot(filter_handler.reshape_coeffs(ZZ)).reshape(im.shape)
sparsity = misc.get_sparsity(ZZ)

Y_l2, Z_l2 = filter_handler.low_pass_filter(im, k)
Y_l2 = Y_l2.reshape(im.shape)
print("l1-norm of coefficients: {}".format(np.abs(Z).sum()))
print("sparsity: {}".format(sparsity))
print("l2 sparsity: {}".format(misc.get_sparsity(Z_l2)))

misc.save_image(im * scaling, join(folder, '{}.png'.format(imName)))
misc.save_image(recons * scaling, join(folder, '{}_{}_sparse_{}_wasserstein.png'.format(imName, filter_type, k)))
misc.save_image(Y_l2 * scaling, join(folder, '{}_{}_sparse_{}_euclidean.png'.format(imName, filter_type, k)))
