from skimage.exposure import equalize_hist, equalize_adapthist
from skimage.io import imread
from skimage.color import rgb2gray
from matplotlib import pyplot as plt

if __name__ == "__main__":
    print("__________Q3 Running__________")
    image_path = "images/chang.tif"
    image = rgb2gray(imread(image_path))
    number_of_bins = 256
    equalized_image = equalize_hist(image, nbins=number_of_bins)

    adapt_equalized_image_default = equalize_adapthist(image)
    kernel_size = [11, 11]
    clip_limit = 0.05
    adapt_equalized_image = equalize_adapthist(image, kernel_size=kernel_size, clip_limit=clip_limit)
    output_path = image_path.replace(image_path.split("/")[-1], "equalize_" + image_path.split("/")[-1])
    print("Plotting image: ", output_path)
    plt.figure(figsize=(16, 12))
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)

    ax1.imshow(image, cmap='gray')
    ax1.set_title("Original", fontsize=15)

    ax2.imshow(equalized_image, cmap='gray')
    ax2.set_title("Equalized(#bins ="+str(number_of_bins)+")", fontsize=15)

    ax3.imshow(adapt_equalized_image_default, cmap='gray')
    ax3.set_title(f"Adaptive Equalized(default)", fontsize=15)

    ax4.imshow(adapt_equalized_image, cmap='gray')
    ax4.set_title(f"Adaptive Equalized(kernel={kernel_size}, clip={clip_limit})", fontsize=15)

    plt.savefig(output_path)
    plt.show()
