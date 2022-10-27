from skimage import color
from matplotlib import pyplot as plt
import numpy as np
from skimage import measure
from q_1_histogram import read_image, get_histogram


def double_sided_thresholding(low_threshold, high_threshold, original_image):
    threshold_image = np.full(original_image.shape, original_image.max())
    indexes = np.where(
        np.logical_and(
            original_image > low_threshold,
            original_image < high_threshold
        )
    )

    for i in range(len(indexes[0])):
        threshold_image[indexes[0][i]][indexes[1][i]] = original_image.min()
    return threshold_image


def find_connected_components(threshold_image, threshold):
    blob_image = threshold_image > threshold
    blobs_labels, count = measure.label(blob_image, return_num=True)
    blobs_labels = color.label2rgb(blobs_labels, bg_label=0)
    return blob_image, blobs_labels


def plot_images(original_image, threshold_image, original_histogram, threshold_histogram, threshold_params, blob_image, blob_labels, path):
    original_bin_edges = original_histogram.T[0]
    original_bin_count = original_histogram.T[1]
    assert len(original_bin_edges) == len(original_bin_count)
    output_path = path.replace(path.split("/")[-1], "double_thresholding_" + path.split("/")[-1])
    print("Plotting image: ", output_path)

    plt.figure(figsize=(20, 15))
    ax1 = plt.subplot(3, 2, 1)
    ax2 = plt.subplot(3, 2, 2)
    ax3 = plt.subplot(3, 2, 3)
    ax4 = plt.subplot(3, 2, 4)
    ax5 = plt.subplot(3, 2, 5)
    ax6 = plt.subplot(3, 2, 6)

    ax1.imshow(image, cmap='gray')
    ax1.set_title("Original Image", fontsize=15)

    if int(original_image.max()) == 1:
        width = 0.05
    else:
        width = 5
    ax2.bar(original_bin_edges, original_bin_count, width=width, align='center', alpha=0.5)
    ax2.grid(axis='y', alpha=0.75)
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Original Histogram(n=' + str(len(original_bin_count)) + ")")
    ax2.set_xticks(original_bin_edges)

    ax3.imshow(threshold_image, cmap='gray')
    ax3.set_title(f"Threshold Image(low={threshold_params[0]}, high={threshold_params[1]})", fontsize=15)

    threshold_bin_edges = threshold_histogram.T[0]
    threshold_bin_count = threshold_histogram.T[1]
    width = 0.05 if int(threshold_image.max()) == 1 else 5
    ax4.bar(threshold_bin_edges, threshold_bin_count, width=width, align='center', alpha=0.5)
    ax4.grid(axis='y', alpha=0.75)
    ax4.set_xlabel('Value')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Threshold Image Histogram(n=' + str(len(threshold_bin_count)) + ")")
    ax4.set_xticks(threshold_bin_edges)

    ax5.imshow(blob_image, cmap='gray')
    ax5.set_title(f"Connected Components(threshold={threshold_params[2]})", fontsize=15)

    ax6.imshow(blob_labels)
    ax6.set_title("Labeled Connected Components", fontsize=15)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


if __name__ == "__main__":
    print("__________Q2 Running__________")
    image_path = "images/turkeys.tif"
    image = read_image(image_path)
    histogram = get_histogram(image, num_of_bins=10)
    low_value, high_value, connected_thresh = 105.0, 187, 43
    thresh_image = double_sided_thresholding(low_value, high_value, image)
    thresh_histogram = get_histogram(thresh_image, num_of_bins=10)
    blobs, labels = find_connected_components(thresh_image, threshold=connected_thresh)
    plot_images(
        path=image_path,
        original_image=image,
        original_histogram=histogram,
        threshold_image=thresh_image,
        threshold_histogram=thresh_histogram,
        threshold_params=[low_value, high_value, connected_thresh],
        blob_image=blobs,
        blob_labels=labels
    )
