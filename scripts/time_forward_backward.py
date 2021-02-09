from os import makedirs
from os.path import join, pardir, exists

import numpy as np
from matplotlib import pyplot as plt

from ot_sparse_projection import ot_sparse_projection, misc, optim
from ot_sparse_projection.dictionaries import get_filter_handler
from ot_sparse_projection.optim import Timer

folder = join(pardir, 'img', 'timing')
if not exists(folder):
    makedirs(folder)
n = 64

fname = join(folder,"forward_backward_timing.npz")
filter_type = 'dct'
gamma = .1
imName = 'racoon'
lamb = .5

im, scaling = misc.get_image(imName, n)
vmin = 0
vmax = im.reshape(-1).max()

filter_handler = get_filter_handler(im, filter_type)
timer_inv = Timer()
im_inv, y_inv, _ = ot_sparse_projection.wasserstein_image_filtering_invertible_dictionary(im, filter_handler, gamma,
                                                                                          lamb,
                                                                                          timer=timer_inv, precision=1e-12,
                                                                                          linesearch_type=optim.DECREASING)
plt.subplot(1, 2, 1)
plt.imshow(im.reshape([n, n]), cmap='gray', vmin=vmin, vmax=vmax)
plt.subplot(1, 2, 2)
plt.imshow(im_inv.reshape([n, n]), cmap='gray', vmin=vmin, vmax=vmax)
plt.show()

timer_ort = Timer()
im_ort, y_ort, _ = ot_sparse_projection.wasserstein_image_filtering_orthogonal_dictionary(im, filter_handler, gamma,
                                                                                          lamb,
                                                                                          linesearch_type=optim.DECREASING,
                                                                                          timer=timer_ort,
                                                                                          precision=1e-10)

times_inv = timer_inv.times
times_ort = timer_ort.times
obj_inv = timer_inv.objectives
obj_ort = timer_ort.objectives
min_val = min(np.min(timer_inv.objectives), np.min(timer_ort.objectives))

timer_fb = Timer()
im_fb, y_fb, _ = ot_sparse_projection.wasserstein_image_projection_forward_backward(im, filter_handler, gamma, lamb,
                                                                                    timer=timer_fb, max_iter=200000)

times_fb = timer_fb.times
obj_fb = timer_fb.objectives
np.savez(fname, im_fb, y_fb, times_fb, obj_fb)

x = np.load(fname)
im_fb, y_fb, times_fb, obj_fb = [x['arr_{}'.format(i)] for i in range(0, 4)]

min_val = np.min([np.min(obj_inv), np.min(obj_ort), np.min(obj_fb)])
plt.loglog(times_inv, obj_inv - min_val, label='Dual method, invertible dictionary')
plt.loglog(times_ort, obj_ort - min_val, label='Dual method, orthonormal dictionary')
plt.loglog(times_fb, obj_fb - min_val, label='Forward backward method')
plt.xlabel('Time (s)')
plt.ylabel('Optimality gap')
plt.legend()
plt.savefig(join(folder, 'forward_backward_times.eps'))
plt.show()
