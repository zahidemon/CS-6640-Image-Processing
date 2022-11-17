from matplotlib import pyplot as plt
from utils import read_image, butterworth_filter, apply_fft_with_shift

if __name__ == "__main__":
    image_paths = [
        "data/cell0.png",
        "data/cell1.png",
        "data/cell2.png",
    ]
    plt.figure(figsize=(20, 25))
    axes = []
    for index, image_path in enumerate(image_paths):
        image = read_image(image_path)
        power_spectrum = apply_fft_with_shift(image)
        filtered_image, _ = butterworth_filter(image)

        axes.append(plt.subplot(len(image_paths), 3, index*3 + 1))
        axes[index*3].imshow(image, cmap="gray")
        axes[index*3].set_title("Original Image", fontsize=15)

        axes.append(plt.subplot(len(image_paths), 3, index * 3 + 2))
        axes[index * 3 + 1].imshow(power_spectrum, cmap="gray")
        axes[index * 3 + 1].set_title("Power Spectrum", fontsize=15)

        axes.append(plt.subplot(len(image_paths), 3, index * 3 + 3))
        axes[index * 3 + 2].imshow(filtered_image, cmap="gray")
        axes[index * 3 + 2].set_title("Filtered Image", fontsize=15)

    output_path = "data/q1_output.jpg"
    print("plotting image: ", output_path)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

