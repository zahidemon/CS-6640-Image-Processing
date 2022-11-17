from skimage.io import imread
from skimage.color import rgb2gray
import numpy as np


def read_image(path):
    image = imread(path)
    if len(image.shape) > 2:
        return rgb2gray(image)
    return image


def apply_fft_with_shift(image, do_log_transform=True):
    img_fft = np.fft.fft2(image)
    img_fft_shift = np.fft.fftshift(img_fft)
    if do_log_transform:
        return np.log(np.abs(img_fft_shift)**2)
    return img_fft_shift


def build_filter(shape, cutoff_freq=50, n=1, shifted=False):
    if shifted:
        c_freq = np.fft.fftshift(np.fft.fftfreq(shape[1]) * shape[1])
        r_freq = np.fft.fftshift(np.fft.fftfreq(shape[0]) * shape[0])
    else:
        c_freq = np.fft.fftfreq(shape[1]) * shape[1]
        r_freq = np.fft.fftfreq(shape[0]) * shape[0]
    a = np.tile(c_freq ** 2, (shape[0], 1))
    b = np.tile((r_freq[:, np.newaxis]) ** 2, (1, shape[1]))
    return 1 / (1 + (np.sqrt(a + b) / cutoff_freq) ** (2 * n))


def butterworth_filter(image, n=1, cutoff_freq=50):
    filter = build_filter(image.shape, cutoff_freq=cutoff_freq, n=n, shifted=True)
    img_fft_shift = apply_fft_with_shift(image, do_log_transform=False)
    butterworth_filter_power_spectrum = np.log(np.abs(filter) ** 2)
    filtered_img = np.fft.ifft2(np.fft.ifftshift(img_fft_shift.copy() * filter))
    return np.real(filtered_img), np.real(butterworth_filter_power_spectrum)


def phase_correlation(image1, image2, cutoff_freq=50, n=1):
    img1_fft = np.fft.fft2(image1)
    img2_fft = np.fft.fft2(image2)
    f_star = np.conjugate(img1_fft)
    g = img2_fft
    h = build_filter(image1.shape, cutoff_freq=cutoff_freq, n=n)
    p = np.fft.ifft2(h*(f_star*g/np.abs(f_star*g)))
    return np.real(p)


def find_peak(image1, image2, cutoff_freq=50, n=1):
    phase_corr = phase_correlation(image1, image2, cutoff_freq=cutoff_freq, n=n)
    peak_idx = np.unravel_index(np.argmax(phase_corr), phase_corr.shape)
    peak_val = phase_corr[peak_idx]
    return peak_idx, peak_val, phase_corr
