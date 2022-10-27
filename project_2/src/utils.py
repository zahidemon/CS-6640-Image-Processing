import random

import cv2
import numpy as np
from skimage.color import rgb2gray
from skimage.io import imread


def read_image(path):
    image = imread(path)
    if len(image.shape) > 2:
        return rgb2gray(image)
    return image


def get_gaussian(image, mean, variance):
    gaussian = np.random.normal(mean, variance, image.shape)
    gaussian = np.reshape(gaussian, image.shape).astype(image.dtype)
    assert gaussian.shape == image.shape
    assert gaussian.dtype == image.dtype
    return gaussian


def add_replacement(image):
    assert len(image.shape) == 2
    height, width = image.shape

    number_of_pixels = random.randint((height*width) // 4, (height*width) // 2)
    for i in range(number_of_pixels):
        white_y = random.randint(0, width - 1)
        white_x = random.randint(0, height - 1)
        image[white_x][white_y] = image.max()

        black_y = random.randint(0, width - 1)
        black_x = random.randint(0, height - 1)
        image[black_x][black_y] = image.min()
    return image


def add_poisson(image):
    return np.random.poisson(image) + image


def add_gaussian(image, mean, variance):
    gaussian = get_gaussian(image, mean, variance)
    return gaussian + image


def add_speckle(image, mean, variance):
    gaussian = get_gaussian(image, mean, variance)
    return image + image * gaussian


def add_noise(image, noise_type):
    if noise_type == "gaussian":
        return add_gaussian(image, 0, 0.25)
    elif noise_type == "speckle":
        return add_speckle(image, 0, 0.25)
    elif noise_type == "poisson":
        return add_poisson(image)
    elif noise_type == "replacement":
        return add_replacement(image)
    else:
        raise NotImplementedError(f"{noise_type} has not been implemented yet.")


def add_padding(image, n):
    for i in range(n):
        image = np.pad(image, 1)
    return image


def get_mean(image, n, h, w, original_h, original_w):
    total, count = 0, 0
    h_start = max(0, h-n)
    h_end = min(h+n+1, original_h)
    w_start = max(0, w-n)
    w_end = min(original_w, w+n+1)
    for i in range(h_start, h_end):
        for j in range(w_start, w_end):
            total += image[i][j]
            count += 1
    if total / count == 0:
        print(n, h, w)
        print(total, count)
    return total / count


def box_filter(image, kernel_size, padding=True):
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number")
    padding_size = kernel_size // 2
    if padding:
        padded_image = add_padding(image, padding_size)
    else:
        padded_image = add_padding(image, 0)
    height, width = image.shape
    print(height, width, padding_size)
    for h in range(padding_size, height):
        for w in range(padding_size, width):
            padded_image[h][w] = get_mean(padded_image, padding_size, h, w, height, width)
    if padding:
        end_height, end_width = padding_size + height, padding_size + width
    else:
        end_height, end_width = height - padding_size, width - padding_size
    return padded_image[padding_size:end_height, padding_size:end_width]


def convolution_filter(image, kernel):
    return cv2.filter2D(image, -1, kernel)


def gaussian_filter(image, kernel):
    return cv2.GaussianBlur(image, kernel, 0)


def calculate_mse(image_1, image_2):
    assert image_1.shape == image_2.shape
    return round(np.square(np.subtract(image_1, image_2)).mean(), 4)


def bilateral_filter(image, diameter, sigma_color, sigma_space):
    print(image.max())
    if image.max() != 255:
        image = image * 255
        image = image.astype('uint8')
        return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space) / 255
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)


def median_filter(image, kernel_size):
    if image.max() != 255:
        image = image * 255
        image = image.astype('uint8')
        return cv2.medianBlur(image, kernel_size) / 255
    return cv2.medianBlur(image, kernel_size)


def non_local_mean_filter(image, h):
    if image.max() != 255:
        image = image * 255
        image = image.astype('uint8')
        return cv2.fastNlMeansDenoising(image, h=h) / 255
    return cv2.fastNlMeansDenoising(image, h=h)
