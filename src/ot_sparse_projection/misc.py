from ot_sparse_projection import shapes
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from pywt import data


def get_image(im_name, n):
    if im_name == 'racoon':
        im = misc.face(True)
    elif im_name == 'square':
        im = shapes.get_square(256, 96)
    elif im_name == 'circle':
        im = shapes.get_circle(256, 96)
    elif im_name == 'diamond':
        im = shapes.get_diamond(256,96)
    elif im_name == 'camera':
        im = data.camera()
    elif im_name == 'aero':
        im = data.aero()
    elif im_name == 'ascent':
        im = data.ascent()
    elif im_name == 'nino':
        im = data.nino()

    else:
        from PIL import Image
        im = misc.imread(im_name)
    im = misc.imresize(im, [n, n]).astype(np.float64)
    scaling = im.sum()
    return im/scaling, scaling


def save_image(x, fname):
    from PIL import Image
    x = x.copy()
    x[x<0] = 0
    x[x>255] = 255
    x = x.astype(np.uint8)
    im = Image.fromarray(x)
    im.save(fname)


def plot_log_hist(hist):
    x = np.sort(np.abs(hist.ravel()))[::-1]
    plt.plot(np.log(x + 1e-30) / np.log(10))

def get_sparsity(X):
    return float((X!=0).sum()) / X.size


def is_power_of_two(a):
    if a<=0:
        return False
    return __is_power_of_two(a)


def __is_power_of_two(a):
    if int(a) != a:
        return False
    if a==1:
        return True
    return is_power_of_two(a/2.)
