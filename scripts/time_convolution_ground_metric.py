import time
from os import makedirs
from os.path import join, pardir, exists

from matplotlib import pyplot as plt

from ot_sparse_projection import misc, ot_ground_metric

folder = join(pardir, 'img', 'timing')
if not exists(folder):
    makedirs(folder)

filter_type = 'db2'
gamma = .1
lamb = 1
imName = 'racoon'

sizes = [4, 8, 12, 16, 24, 32, 48, 64, 96, 110]
sizes_convolution = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 424, 512,
                     768, 1024]
# sizes = [8, 16, 32]
convolution_times = []
standard_times = []


def time_projection(n, type):
    print(type)
    print(n)
    im, scaling = misc.get_image(imName, n)
    D = ot_ground_metric.get_handler(type, im.shape, gamma)
    im = im.ravel()
    t = time.time()
    print(t)
    D.dot(im)
    print(time.time())
    return time.time() - t


for n in sizes_convolution:
    convolution_times.append(time_projection(n, ot_ground_metric.CONVOLUTION))
for n in sizes:
    standard_times.append(time_projection(n, ot_ground_metric.MATRIX))

plt.loglog(sizes, standard_times, label='Matrix multiplication', marker='D', basex=2)
plt.loglog(sizes_convolution, convolution_times, label='Convolution acceleration', marker='o', basex=2)
plt.legend()
plt.xlabel('Image pixel width')
plt.ylabel('Computing time')
plt.savefig(join(folder, 'convolution_timing.eps'))
plt.show()
