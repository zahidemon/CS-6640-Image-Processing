from skimage.io import imread
from skimage.color import rgb2gray
from skimage.util import img_as_float
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


def load_images_from_txt(directory_path, txt_filename):
    cells_txt_filepath = directory_path + txt_filename
    image_dict = {}
    with open(cells_txt_filepath, 'r') as f:
        for line in f:
            image_filename = line.rstrip('\n')
            img = img_as_float(imread(directory_path+image_filename))
            image_dict[image_filename] = img
    return image_dict


def phase_correlation(image1, image2, cutoff_freq=50, n=1):
    img1_fft = np.fft.fft2(image1)
    img2_fft = np.fft.fft2(image2)
    f_star = np.conjugate(img1_fft)
    g = img2_fft
    h = build_filter(image1.shape, cutoff_freq=cutoff_freq, n=n)
    p = np.fft.ifft2(h*(f_star*g/np.abs(f_star*g)))
    return np.real(p)


def find_peak(image1, image2, threshold=0.01, cutoff_freq=50, n=1, check_overlap=True):
    phase_corr = phase_correlation(image1, image2, cutoff_freq=cutoff_freq, n=n)
    peak_idx = np.unravel_index(np.argmax(phase_corr), phase_corr.shape)
    peak_val = phase_corr[peak_idx]
    if check_overlap:
        if peak_val > threshold:
            return peak_idx, peak_val, phase_corr
        else:
            return None
    else:
        return peak_idx, peak_val, phase_corr


def get_best_correlated_image(image_dict, anchor_img, active_set, thresh, cutoff_freq=50, n=1):
    thresh_met = False
    max_shift = (0, 0)
    max_corr = 0
    max_corr_img_key = ''
    for img_key in image_dict:
        if img_key in active_set:
            continue
        img = image_dict[img_key]
        fp_output = find_peak(img, anchor_img, threshold=thresh, cutoff_freq=cutoff_freq, n=n)
        if fp_output is None:
            continue
        else:
            thresh_met = True
        shift, corr, pc = fp_output
        if corr > max_corr:
            max_corr = corr
            max_shift = shift
            max_corr_img_key = img_key
    if thresh_met:
        return max_corr_img_key, max_shift, max_corr
    else:
        return None


def create_mosaic_dict_list(image_dict, thresh=0.01, cutoff_freq=50, n=1):
    image_key_list = list(image_dict.keys())
    active_set = {image_key_list[0]}
    mosaic_dict_list = {}
    img_keys_to_anchor = [image_key_list[0]]

    while len(img_keys_to_anchor) > 0:
        anchor_img_key = img_keys_to_anchor.pop()
        while True:
            res = get_best_correlated_image(image_dict, image_dict[anchor_img_key], active_set, thresh,
                                            cutoff_freq=cutoff_freq, n=n)
            if res is None:
                break
            else:
                max_corr_img_key, max_shift, max_corr = res
                active_set.add(max_corr_img_key)
                img_keys_to_anchor.append(max_corr_img_key)
                if anchor_img_key in mosaic_dict_list:
                    mosaic_dict_list[anchor_img_key].append((max_corr_img_key, max_shift, max_corr))
                else:
                    mosaic_dict_list[anchor_img_key] = [(max_corr_img_key, max_shift, max_corr)]

    image_keys_not_used = set(image_dict.keys()) - active_set
    if len(image_keys_not_used) > 0:
        print('Did not use the following keys (never met any pairing threshold) {0}'.format(image_keys_not_used))
    return mosaic_dict_list


def create_mosaic(image_dict, thresh=0.01):
    mosaic_dict_list = create_mosaic_dict_list(image_dict, thresh=thresh, cutoff_freq=50, n=1)

    image_key_list = list(image_dict.keys())
    image_list_length = len(image_key_list)
    image_size_r, image_size_c = image_dict[image_key_list[0]].shape

    blank_canvas = np.zeros(
        (2 * image_size_r * (image_list_length + 1), 2 * image_size_c * (image_list_length + 1)))
    canvas = blank_canvas.copy()

    current_anchor_r = image_size_r * image_list_length
    current_anchor_c = image_size_c * image_list_length
    slice_dict = {}
    for anchor_img_key in mosaic_dict_list:
        if anchor_img_key in slice_dict:
            current_anchor_r = slice_dict[anchor_img_key][0].start
            current_anchor_c = slice_dict[anchor_img_key][1].start

        anchor_img = image_dict[anchor_img_key]
        ai_r = anchor_img.shape[0]
        ai_c = anchor_img.shape[1]
        ai_slice = (slice(current_anchor_r, current_anchor_r + ai_r), slice(current_anchor_c, current_anchor_c + ai_c))

        canvas[ai_slice] = anchor_img
        anchor_list = mosaic_dict_list[anchor_img_key]
        for anchor_tup in anchor_list:
            max_corr_img_key, max_shift, max_corr = anchor_tup
            max_corr_img = image_dict[max_corr_img_key]

            mc_r = max_corr_img.shape[0]
            mc_c = max_corr_img.shape[1]
            max_shift_r = max_shift[0]
            max_shift_c = max_shift[1]

            case1_slice = (slice(current_anchor_r + max_shift_r, current_anchor_r + max_shift_r + mc_r),
                           slice(current_anchor_c + max_shift_c, current_anchor_c + max_shift_c + mc_c))
            case2_slice = (slice(current_anchor_r + max_shift_r - mc_r, current_anchor_r + max_shift_r),
                           slice(current_anchor_c + max_shift_c - mc_c, current_anchor_c + max_shift_c))
            case3_slice = (slice(current_anchor_r + max_shift_r - mc_r, current_anchor_r + max_shift_r),
                           slice(current_anchor_c + max_shift_c, current_anchor_c + max_shift_c + mc_c))
            case4_slice = (slice(current_anchor_r + max_shift_r, current_anchor_r + max_shift_r + mc_r),
                           slice(current_anchor_c + max_shift_c - mc_c, current_anchor_c + max_shift_c))
            slice_list = [case1_slice, case2_slice, case3_slice, case4_slice]

            best_slice = case1_slice
            best_corr = 0.0
            for case_slice in slice_list:
                case = blank_canvas.copy()
                case[case_slice] = max_corr_img

                fp_output = find_peak(case[ai_slice], anchor_img, threshold=0, cutoff_freq=50, n=1)

                if fp_output is None:
                    continue
                _, case_corr, pc = fp_output

                if case_corr > best_corr:
                    best_corr = case_corr
                    best_slice = case_slice
            canvas[best_slice] = max_corr_img
            slice_dict[max_corr_img_key] = best_slice

    zero_idx_c = np.argwhere(np.all(canvas == 0.0, axis=0))
    canvas = np.delete(canvas, zero_idx_c, axis=1)
    zero_idx_r = np.argwhere(np.all(canvas == 0.0, axis=1))
    canvas = np.delete(canvas, zero_idx_r, axis=0)
    return canvas
