import math

import numpy as np
import pywt
from scipy import special

from . import proximal


class Shrinker(object):

    def __init__(self, im, wavelet, max_level=None, *args, **kwargs):
        self.im = im
        self.wavelet = wavelet
        self.thresholds = None
        if max_level is None:
            max_level = pywt.dwt_max_level(im.shape[0], wavelet.type)
        self.max_level = max_level
        self.bands = self.wavelet.dec(self.im, level=self.max_level)
        self.sigma = np.median(np.abs(self.bands[-1][-1]).ravel()) / 0.6745

    def compute_threshold(self):
        thresholds = [np.zeros_like(self.bands[0])]
        for level in range(1, self.max_level + 1):
            thresholds.append(self.compute_level_threshold(level))
        self.thresholds = self.wavelet.coeffs_to_array(thresholds)
        return self.thresholds

    def compute_level_threshold(self, level):
        raise NotImplementedError()

    def get_mask(self):
        return (self.get_threshold() > 0).astype(int)

    def get_threshold(self):
        if self.thresholds is None:
            self.compute_threshold()
        return self.thresholds

    def denoise(self):
        if self.thresholds is None:
            self.compute_threshold()
        y = self.wavelet.inverse_dot(self.im)
        y = proximal.prox_l1(y, self.thresholds)
        return self.wavelet.dot(y).reshape(self.im.shape), y


class NormalShrink(Shrinker):

    def compute_level_threshold(self, level):
        bands = self.bands[level]
        LH = bands[0]
        HL = bands[1]
        HH = bands[2]
        beta = (math.log((HH.size + HL.size + LH.size) / self.max_level) ** (1. / 2))
        return [self.compute_subband_threshold(band, beta) for band in bands]

    def compute_subband_threshold(self, subband, beta):
        sigma_y = np.std(subband.ravel())
        return beta * (self.sigma ** 2.) * np.ones_like(subband) / sigma_y

    def get_mask(self):
        return (self.get_threshold() > 0).astype(int)

    def get_threshold(self):
        if self.thresholds is None:
            self.compute_threshold()
        return self.thresholds


class Shrinker2016(Shrinker):
    THRESHOLD = 0.2

    def __init__(self, im, wavelet, *args, **kwargs):
        super().__init__(im, wavelet, *args, **kwargs)
        self.s = None
        self.low_thresholds = None
        self.high_thresholds = None
        self.k = None

    def compute_threshold(self):
        if self.s is None:
            self.compute_peak_to_sums()
        self.low_thresholds = []
        self.high_thresholds = []
        peak_to_sum_high_ref = self.compute_peak_to_sum_ref(1)
        peak_to_sum_low_ref = self.compute_peak_to_sum_ref(-1)
        for level in range(1, self.k + 1):
            bands = self.wavelet.dec(self.im, level=level)
            detail = np.concatenate(bands[1]).ravel()
            self.compute_detail_threshold(detail, peak_to_sum_high_ref, peak_to_sum_low_ref, self.s[level],
                                          shapes=[bands[1][0].shape,
                                                  bands[1][1].shape,
                                                  bands[1][2].shape])

        self.compute_detail_threshold(bands[0], peak_to_sum_high_ref, peak_to_sum_low_ref, 1)
        self.high_thresholds = self.wavelet.coeffs_to_array(self.high_thresholds)
        self.low_thresholds = self.wavelet.coeffs_to_array(self.low_thresholds)
        self.thresholds = (self.high_thresholds + self.low_thresholds) / 2
        return self.thresholds

    def denoise(self):
        if self.low_thresholds is None or self.high_thresholds is None:
            self.compute_threshold()
        y = self.wavelet.inverse_dot(self.im)
        y[np.logical_and(y > self.low_thresholds, y < self.high_thresholds)] = 0
        return self.wavelet.dot(y).reshape(self.im.shape), y

    def compute_peak_to_sums(self):
        self.s = []
        self.k = None
        for level in range(1, self.max_level + 1):
            self.s.append(self.compute_level_peak_to_sums(level))
            if self.k is None and self.s[-1] > self.THRESHOLD:
                self.k = level - 1
        if self.k is None:
            self.k = self.max_level
        self.s = self.s[::-1]
        return self.s

    def compute_level_peak_to_sums(self, level, high_low=0):
        return self.compute_detail_peak_to_sums(np.concatenate(self.bands[level][:]), high_low=high_low).ravel()

    def compute_detail_peak_to_sums(self, detail, high_low=0):
        detail = self.get_detail_high_low(detail, high_low)
        s = np.abs(detail)
        value = s.max() / s.sum()
        return 0 if math.isnan(value) else value

    def compute_peak_to_sum_ref(self, high_low):
        s_k = self.compute_level_peak_to_sums(self.k, high_low=high_low)
        s_k1 = self.compute_level_peak_to_sums(self.k + 1, high_low=high_low)
        return (s_k + s_k1) / 2

    def compute_kappa(self, detail, peak_to_sum_ref, mu, sigma, high_low):
        if high_low > 0:
            kappa_min = (self.get_detail_high_low(detail, high_low).max() - mu) / sigma
        else:
            kappa_min = (self.get_detail_high_low(detail, high_low).max() + mu) / sigma
        return kappa_min * (
                peak_to_sum_ref - self.compute_detail_peak_to_sums(detail, high_low=high_low)) / peak_to_sum_ref

    def get_detail_high_low(self, detail, high_low):
        if high_low != 0:
            high_low = high_low / math.fabs(high_low)
            assert high_low == 1. or high_low == -1
            detail = detail * high_low
            detail = detail[detail > 0]
            if detail.size == 0:
                detail = np.asarray([0])
        return detail

    def compute_detail_threshold(self, detail, peak_to_sum_high_ref, peak_to_sum_low_ref, s, shapes=None):
        if s < 0.01:
            h = math.inf
            l = math.inf
        else:
            mu = detail.mean()
            sigma = detail.std()
            kappa_high = self.compute_kappa(detail, peak_to_sum_high_ref, mu, sigma, 1)
            kappa_low = self.compute_kappa(detail, peak_to_sum_low_ref, mu, sigma, -1)
            h = mu + kappa_high * sigma
            l = mu - kappa_low * sigma
        if shapes is None:
            self.low_thresholds.insert(0, l * np.ones_like(detail))
            self.high_thresholds.insert(0, h * np.ones_like(detail))
        else:
            self.low_thresholds.insert(0, [l * np.ones(shape) for shape in shapes])
            self.high_thresholds.insert(0, [h * np.ones(shape) for shape in shapes])


class VisuShrink(Shrinker):

    def compute_threshold(self):
        thresholds = []
        bands = [np.ones([1, 1])]
        for level in range(1, self.max_level + 1):
            bands = self.wavelet.dec(self.im, level=level)
            thresholds.insert(0, self.compute_level_threshold(bands))
        thresholds.insert(0, np.zeros_like(bands[0]))
        self.thresholds = self.wavelet.coeffs_to_array(thresholds)
        return self.thresholds

    def compute_level_threshold(self, bands):
        value = self.sigma * math.sqrt(2 * np.log(self.im.shape[0]))
        return [np.ones_like(band) * value for band in bands[1]]


class BayesShrink(Shrinker):


    def compute_level_threshold(self, level):
        return [self.compute_band_threshold(band) for band in self.bands[level]]

    def compute_band_threshold(self, band):
        sigma_y = np.std(band)
        if sigma_y <= self.sigma:
            value = np.abs(band).max() + 1e-15
        else:
            sigma_x = math.sqrt(np.max(sigma_y ** 2. - self.sigma ** 2., 0))
            value = (self.sigma ** 2.) / sigma_x
        return np.ones_like(band) * value


class SureShrink(Shrinker):

    def __init__(self, im, wavelet, max_level=None, *args, **kwargs):
        super().__init__(im, wavelet, max_level=max_level, *args, **kwargs)
        self.visu = VisuShrink(im, wavelet, max_level=max_level)

    def compute_level_threshold(self, level):
        return [self.compute_band_threshold(band) for band in self.bands[level]]

    def compute_band_threshold(self, band):
        x = np.sort(np.abs(band).ravel())
        xx = x ** 2.
        cum = np.cumsum(xx)
        d = x.size
        indices = np.arange(d)
        sure = d * self.sigma ** 2. - 2 * (indices + 1) * self.sigma ** 2. + cum + indices[::-1] * xx
        index = np.argmin(sure)
        return x[index] * np.ones_like(band)

    def compute_threshold(self):
        self.thresholds = np.minimum(super().compute_threshold(), self.visu.get_threshold())
        return self.thresholds


class NewThresh(VisuShrink):


    def __init__(self, im, wavelet, max_level=None, alpha=.1, *args, **kwargs):
        super().__init__(im, wavelet, max_level=max_level, *args, **kwargs)
        self.alpha = alpha

    def denoise(self):
        self.get_threshold()
        y = self.wavelet.inverse_dot(self.im)
        y[np.abs(y)<=self.thresholds] = 0
        y[y>self.thresholds] = self.threshold_positive(y, y>self.thresholds)
        y[y<-self.thresholds] = self.threshold_negative(y, y<-self.thresholds)
        return self.wavelet.dot(y).reshape(self.im.shape), y

    def threshold_positive(self, y, indices):
        y = y[indices]
        thresholds = self.thresholds[indices]
        return y - thresholds + 2*thresholds/math.pi * special.erf(self.alpha*(y-thresholds)/thresholds)

    def threshold_negative(self, y, indices):
        y = y[indices]
        thresholds = self.thresholds[indices]
        return y + thresholds - 2*thresholds/math.pi * special.erf(-self.alpha*(y+thresholds)/thresholds)