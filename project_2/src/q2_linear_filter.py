import numpy as np
from matplotlib import pyplot as plt
from utils import read_image, box_filter, add_noise, calculate_mse, convolution_filter, gaussian_filter

if __name__ == "__main__":
    image_path = "images/xray.png"
    image = read_image(image_path)
    noise_type = "replacement"

    noisy_image = add_noise(image.copy(), noise_type=noise_type)
    box_filtered_image = box_filter(noisy_image, kernel_size=5, padding=True)
    kernel = np.ones((10, 10), np.float32) / 100
    conv_filtered_image = convolution_filter(image, kernel=kernel)
    gaussian_filtered_image = gaussian_filter(image, kernel=(3, 3))

    noisy_mse = calculate_mse(image, noisy_image)
    conv_mse = calculate_mse(image, conv_filtered_image)
    box_mse = calculate_mse(image, box_filtered_image)
    gaussian_mse = calculate_mse(image, gaussian_filtered_image)

    plt.figure(figsize=(16, 10))
    ax1 = plt.subplot(3, 2, 1)
    ax1.imshow(image, cmap="gray")
    ax1.set_title("Original Image")

    ax2 = plt.subplot(3, 2, 2)
    ax2.imshow(noisy_image, cmap="gray")
    ax2.set_title(f"Noisy Image({noise_type})")

    ax3 = plt.subplot(3, 2, 3)
    ax3.imshow(conv_filtered_image, cmap="gray")
    ax3.set_title(f"Filtered Image-conv filter-MSE improve:{noisy_mse-conv_mse})")

    ax4 = plt.subplot(3, 2, 4)
    ax4.imshow(box_filtered_image, cmap="gray")
    ax4.set_title(f"Filtered Image-box filter-MSE improve:{noisy_mse-box_mse})")

    ax5 = plt.subplot(3, 2, 5)
    ax5.imshow(box_filtered_image, cmap="gray")
    ax5.set_title(f"Filtered Image-gaussian filter-MSE improve:{noisy_mse-gaussian_mse})")

    output_path = image_path.replace(image_path.split("/")[-1], f"output_filter_{noise_type}" + image_path.split("/")[-1])
    print("plotting image: ", output_path)
    plt.savefig(output_path)
    plt.show()


