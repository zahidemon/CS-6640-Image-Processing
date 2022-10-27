from matplotlib import pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray


def read_image(path):
    return rgb2gray(imread(path))


def get_histogram(image, num_of_bins: int):
    height, width = image.shape
    # getting max and min pixel value in the image
    max_pixel_value = image.max()
    min_pixel_value = image.min()
    print("Maximum pixel value in the image: ", max_pixel_value)
    print("Minimum pixel value in the image: ", min_pixel_value)

    # getting the bin edges using numpy built in linear space function
    bin_edges = np.linspace(min_pixel_value, max_pixel_value, num_of_bins + 1, True)
    bin_count = np.zeros(num_of_bins)

    for bin_number in range(len(bin_edges) - 1):
        indexes = np.where(
            np.logical_and(
                image >= bin_edges[bin_number],
                image < bin_edges[bin_number+1]
            )
        )
        bin_count[bin_number] = len(indexes[0])
    bin_count[-1] += len(np.where(image == bin_edges[-1])[0])
    print("Bin edges: ", bin_edges)
    print("Bin counts: ", bin_count)

    # check if all pixels are processed
    assert height*width == sum(bin_count)

    return np.vstack((bin_edges[:-1], bin_count)).T


def plot_histogram(bin_edges, bin_count, path):
    assert len(bin_edges) == len(bin_count)
    output_path = path.replace(path.split("/")[-1], "output_"+str(len(bin_count))+"_"+path.split("/")[-1])
    print("Plotting image: ", output_path)
    image = read_image(path)
    width = 0.01 if int(image.max()) == 1 or len(bin_count) == 256 else 5
    bin_edge_indexes = np.linspace(0, len(bin_count)-1, min(len(bin_count)-1, 10), endpoint=True, dtype=int)
    plt.figure(figsize=(16, 6))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    ax1.imshow(image, cmap='gray')
    ax1.set_title("Image", fontsize=15)
    ax2.bar(bin_edges, bin_count, width=width, align='center', alpha=0.5)
    ax2.grid(axis='y', alpha=0.75)
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Histogram(n=' + str(len(bin_count))+")")
    ax2.set_xticks([bin_edges[i] for i in bin_edge_indexes])
    plt.savefig(output_path)
    plt.show()


if __name__ == "__main__":
    print("__________Q1 Running__________")
    image_path = "images/airplane.jpg"
    original_image = read_image(image_path)
    bins = 256
    histogram = get_histogram(original_image, bins)
    print(histogram)
    plot_histogram(
        bin_edges=histogram.T[0],
        bin_count=histogram.T[1],
        path=image_path
    )
