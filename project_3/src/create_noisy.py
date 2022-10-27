import csv
import os
import random
import pandas as pd
from skimage import io
from skimage.util import  random_noise
import numpy as np


def write_csv(csv_file_name, training_dict_list):
    csv_columns = ["RefImageName", "NoiseType", "NoisyImage"]
    try:
        with open(csv_file_name, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in training_dict_list:
                writer.writerow(data)
    except IOError:
        print("I/O error")


def get_gaussian(image, mean, variance):
    gaussian = np.random.normal(mean, variance, image.shape)
    gaussian = np.reshape(gaussian, image.shape).astype(image.dtype)
    assert gaussian.shape == image.shape
    assert gaussian.dtype == image.dtype
    return gaussian


def add_replacement(image, intensity_var):
    return random_noise(image, mode='s&p', amount=intensity_var)


def add_poisson(image):
    return np.random.poisson(image) + image


def add_gaussian(image, mean, variance):
    gaussian = get_gaussian(image, mean, variance)
    return gaussian + image


def add_speckle(image, mean, variance):
    gaussian = get_gaussian(image, mean, variance)
    return image + image * gaussian


def add_noise(image, noise_indicator):
    """
    image is of size m x n with values in the range (0,1).
    noise_indicator is type of noise that needs to be added to the image
    noise_indicator == 0 indicates an addition of Gaussian noise with mean 0 and var 0.08
    noise_indicator == 1 indicates an addtion of salt and pepper noise with intensity variation of 0.08
    noise_indicator == 2 indicates an addition of Poisson noise
    noise_indicator == 3 indicates an addition of speckle noise of mean 0 and var 0.05

    This function should return a noisy version of the input image
    """
    if noise_indicator == 0:
        noisy = add_gaussian(image, mean=0, variance=0.08)
    elif noise_indicator == 1:
        noisy = add_replacement(image, intensity_var=0.08)
    elif noise_indicator == 2:
        noisy = add_poisson(image)
    elif noise_indicator == 3:
        noisy = add_speckle(image, mean=0, variance=0.05)
    else:
        raise NotImplementedError(f"Invalid noise indicator({noise_indicator}). It should be between 0 to 3.")

    return noisy


def main(directory, train, num_of_samples, noise_indicator_low, noise_indicator_high):
    """
    Main driver function for noise generator
    """
    if train == 1:
        name_csv = pd.read_csv(directory + "file_name_train.csv")
        csv_file_name = directory + "../training.csv"
        directory_name = directory + "../training/"
        training_dict_list = [dict() for x in range(num_of_samples)]
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
    else:
        name_csv = pd.read_csv(directory + "file_name_test.csv")
        csv_file_name = directory + "../testing.csv"
        directory_name = directory + "../testing/"
        training_dict_list = [dict() for x in range(num_of_samples)]
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)

    for i in range(num_of_samples):
        dataset_idx = random.randint(
            0, len(name_csv) - 1
        )  # Choose an image randomly from the dataset
        # Read the image from a path
        img_name = os.path.join(directory, name_csv.iloc[dataset_idx, 0])
        image = io.imread(img_name)

        # Normalize the image to range (0,1)
        max_pixel_value = np.max(image)
        image = image / max_pixel_value

        # Choosing the noise randomly
        noise_type = random.randint(noise_indicator_low, noise_indicator_high)
        noisy_image = add_noise(image, noise_type)
        training_dict_list[i] = {
            "RefImageName": img_name,
            "NoiseType": noise_type,
            "NoisyImage": str(i) + ".png",
        }
        io.imsave(directory_name + str(i) + ".png", (noisy_image*255.0).astype(np.uint8))
    write_csv(csv_file_name, training_dict_list)


if __name__ == "__main__":
    main("../Data/nn_data/cats/raw/", 1, 800, 0, 3)  ## creating 800 Samples of Training Data
    main("../Data/nn_data/cats/raw/", 0, 400, 0, 3)  ## creating 400 Samples of Testing Data
    main(
        "../Data/nn_data/pokemon/raw/", 1, 800, 0, 3
    )  ## creating 800 Samples of Training Data
    main(
        "../Data/nn_data/pokemon/raw/", 0, 400, 0, 3
    )  ## creating 400 Samples of Testing Data
