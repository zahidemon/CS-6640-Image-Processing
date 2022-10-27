from matplotlib import pyplot as plt
from utils import read_image, add_noise, calculate_mse, bilateral_filter, median_filter, non_local_mean_filter

if __name__ == "__main__":
    image_path = "images/cartoon_1.jpg"
    image = read_image(image_path)
    noise_type = "gaussian"

    noisy_image = add_noise(image.copy(), noise_type=noise_type)
    bi_filtered_image = bilateral_filter(noisy_image, 15, 75, 75)
    median_filtered_image = median_filter(image, kernel_size=3)
    non_local_filtered_image = non_local_mean_filter(image, h=3)

    noisy_mse = calculate_mse(image, noisy_image)
    bi_mse = calculate_mse(image, bi_filtered_image)
    median_mse = calculate_mse(image, median_filtered_image)
    non_local_mse = calculate_mse(image, non_local_filtered_image)

    plt.figure(figsize=(16, 10))
    ax1 = plt.subplot(3, 2, 1)
    ax1.imshow(image, cmap="gray")
    ax1.set_title("Original Image")

    ax2 = plt.subplot(3, 2, 2)
    ax2.imshow(noisy_image, cmap="gray")
    ax2.set_title(f"Noisy Image({noise_type})")

    ax3 = plt.subplot(3, 2, 3)
    ax3.imshow(bi_filtered_image, cmap="gray")
    ax3.set_title(f"Filtered Image-bilateral filter-MSE improve:{noisy_mse-bi_mse})")

    ax4 = plt.subplot(3, 2, 4)
    ax4.imshow(median_filtered_image, cmap="gray")
    ax4.set_title(f"Filtered Image-median filter-MSE improve:{noisy_mse-median_mse})")

    ax5 = plt.subplot(3, 2, 5)
    ax5.imshow(non_local_filtered_image, cmap="gray")
    ax5.set_title(f"Filtered Image-non local mean filter-MSE improve:{noisy_mse-non_local_mse})")

    output_path = image_path.replace(image_path.split("/")[-1], f"output_non_linear_filter_{noise_type}_" + image_path.split("/")[-1])
    print("plotting image: ", output_path)
    plt.savefig(output_path)
    plt.show()
