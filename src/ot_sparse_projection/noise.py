import numpy as np
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
from skimage.metrics import structural_similarity as similarity
import math

GAUSSIAN = 'gaussian'
SALT_PEPPER = 'salt_pepper'
POISSON = 'poisson'


def add_gaussian(x, level):
    return x + np.random.normal(0, level * x.mean(), x.shape)


def add_salt_pepper(x, level, max_val):
    mask = np.random.rand(*(x.shape))
    x = x.copy()
    x[mask<level] = 0
    x[mask>1-level] = max_val
    return x


def lee_filter(x, size):
    img_mean = uniform_filter(x, (size, size))
    img_sq_mean = uniform_filter(x**2, (size, size))
    img_var = img_sq_mean-img_mean**2

    overall_var = variance(x)
    img_weights = img_var/(img_var+overall_var)
    return img_mean + img_weights*(x-img_mean)


def poisson(x, level, max_val):
    level = level/max_val
    noise = np.random.poisson(x*level)/level
    x = x.copy()+noise
    x[x>max_val] = max_val
    x[x<0] = 0
    return x


def add_noise(x, noise_type, level, max_val=255):
    if noise_type == GAUSSIAN:
        return add_gaussian(x, level)
    if noise_type == SALT_PEPPER:
        return add_salt_pepper(x, level, max_val)
    if noise_type == POISSON:
        return poisson(x, level, max_val)
    raise ValueError("Only gaussian and salt and pepper noises are implemented")


def psnr(img1, img2, scaling=1.):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0/scaling
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def ssim(img1, img2, scaling=1.):
    return similarity(img1*scaling, img2*scaling)