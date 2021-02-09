from os import makedirs
from os.path import join, pardir, exists

import numpy as np

from ot_sparse_projection import misc, ot_sparse_projection, l2
from ot_sparse_projection.dictionaries import get_filter_handler

filter_type = 'dct'
gamma = .1
n = 256
lamb = 2.5
imName = 'ascent'
folder = join(pardir, 'img/compression')

if not exists(folder):
    makedirs(folder)

im, scaling = misc.get_image(imName, n)

filter_handler = get_filter_handler(im, filter_type)

Y, Z, obj = ot_sparse_projection.wasserstein_image_filtering_invertible_dictionary(im, filter_handler, gamma, lamb)
sparsity_pattern = np.not_equal(0, Z)
_, Z_wasserstein_hard, obj_hard = ot_sparse_projection.OtFilteringSpecificPattern(filter_handler, gamma, sparsity_pattern
                                                                                  ).projection(im)
sparsity = misc.get_sparsity(Z)
Y_l2, Z_l2 = l2.sparse_projection(im, filter_handler, sparsity)
Y_l2_hard, Z_l2_hard = l2.hard_thresholding(im, filter_handler, sparsity)

recons = filter_handler.dot(filter_handler.reshape_coeffs(Z)).reshape(im.shape)
recons_hard = filter_handler.dot(filter_handler.reshape_coeffs(Z_wasserstein_hard)).reshape(im.shape)

print("l1-norm of coefficients: {}".format(np.abs(Z).sum()))
print("sparsity: {}".format(sparsity))
print("l2 sparsity: {}".format(misc.get_sparsity(Z_l2)))
print("l2-hard sparsity: {}".format(misc.get_sparsity(Z_l2_hard)))

misc.save_image(im * scaling, join(folder, '{}.png'.format(imName)))
misc.save_image(recons * scaling, join(folder, '{}_{}_sparse_{}_wasserstein.png'.format(imName, filter_type,
                                                                                        int(sparsity * 100))))
misc.save_image(recons_hard * scaling, join(folder, '{}_{}_sparse_{}_wasserstein_hard.png'.format(imName, filter_type,
                                                                                                  int(sparsity * 100))))
misc.save_image(Y_l2_hard * scaling, join(folder, '{}_{}_sparse_{}_euclidean_hard.png'.format(imName, filter_type,
                                                                                              int(sparsity * 100))))
misc.save_image(Y_l2 * scaling, join(folder, '{}_{}_sparse_{}_euclidean.png'.format(imName, filter_type,
                                                                                    int(sparsity * 100))))
