# Goal
***
This repository includes code for performing optimal transport coefficient shrinkage, as described in

* Antoine Rolet and Vivien Seguy. Fast optimal transport regularized projection and application to coefficient shrinkage
  and filtering. The Visual Computer, 2021. doi:
  10.1007/s00371-020-02029-7.

It is maintained by the authors of this paper, and also includes all necessary code to reproduce the experiments
therein.

# Installation
***
Download the source code, then in a shell move to the base directory and run

`pip3 install .`

# Usage
***

This following simple script is from the file in 'script/example.py'
```
import matplotlib.pyplot as plt
from ot_sparse_coding import misc, ot_sparse_coding, l2
from ot_sparse_coding.dictionaries import get_filter_handler
import numpy as np

# Choose the type of wavelet of fourier decomposition you want
filter_type = 'dct'

# Optimal transport regularization strength. Use .1 unless you know better
gamma = .1

n = 256         # Used to resize the image to n x n pixels
lamb = 2.5      # l1 regularization strength. Higher values increase sparsity
imName = 'ascent'       # Path to your image, or name of a pre-configured image

im, scaling = misc.get_image(imName, n)     # get your image, with a rescaling which is usefull to always keep
                                            # similar regularization values
filter_handler = get_filter_handler(im, filter_type)    # handler for Fourier or wavelet transforms

Y, Z, obj = ot_sparse_coding.wasserstein_image_filtering_invertible_dictionary(im, 
                                filter_handler, gamma, lamb, power=2.)  # this computes the optimal transport coefficient shrinkage
sparsity_pattern = np.not_equal(0, Z)
_, Z_wasserstein_hard, obj_hard = ot_sparse_coding.OtFilteringSpecificPattern(filter_handler, gamma, sparsity_pattern,
                                                        power=2.0).projection(im) # this computes the optimal transport hard thresholding
sparsity = misc.get_sparsity(Z)
Y_l2, Z_l2 = l2.sparse_projection(im, filter_handler, sparsity)     # Euclidean coefficient shrinkage with same sparsity
Y_l2_hard, Z_l2_hard = l2.hard_thresholding(im, filter_handler, sparsity) # Euclidean hard thresholding


print(Y.sum())
recons = filter_handler.dot(filter_handler.reshape_coeffs(Z)).reshape(im.shape)
recons_hard = filter_handler.dot(filter_handler.reshape_coeffs(Z_wasserstein_hard)).reshape(im.shape)


print("l1-norm of coefficients: {}".format(np.abs(Z).sum()))
print("sparsity: {}".format(sparsity))
print("l2 sparsity: {}".format(misc.get_sparsity(Z_l2)))
print("l2-hard sparsity: {}".format(misc.get_sparsity(Z_l2_hard)))

# Plot the reconstructed images
vmin = 0
vmax = im.reshape(-1).max()
ax = plt.subplot(2, 2, 1)
plt.imshow(im, cmap='gray', vmin=vmin, vmax=vmax)
ax.set_title('Original image')

ax = plt.subplot(2, 2, 2)
plt.imshow(Y_l2_hard, cmap='gray', vmin=vmin, vmax=vmax)
ax.set_title('Coefficient shrinkage')

ax = plt.subplot(2, 2, 3)
plt.imshow(recons_hard, cmap='gray', vmin=vmin, vmax=vmax)
ax.set_title('OT hard thresholding')

ax = plt.subplot(2, 2, 4)
plt.imshow(recons, cmap='gray', vmin=vmin, vmax=vmax)
ax.set_title('OT coefficient shrinkage')

plt.show()
```

## Supported wavelets
 The supported values for `filter_type` is 'dct' for the discrete Fourier transform, or any of the supported wavelets of 
 the package py_wavelets. The code has been tested for 'dct', 'haar', 'db*' and 'bior*' (replace * with adequate values).