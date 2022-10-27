import numpy as np
from matplotlib import pyplot as plt

weights_from_1c = np.array([0.3, 0.6, 0.1])


def convert_to_grayscale(image):
    if len(image.shape) == 2:
        return image
    return np.dot(image[..., :3], weights_from_1c)


# get gray images
image_1 = plt.imread("images/airplane.jpg")
image_1_gray = convert_to_grayscale(image_1)

image_2 = plt.imread("images/houndog1.png")
image_2_gray = convert_to_grayscale(image_2)

image_3 = plt.imread("images/turkeys.tif")
image_3_gray = convert_to_grayscale(image_3)

image_4 = plt.imread("images/xray.png")
image_4_gray = convert_to_grayscale(image_4)

# plot the images
fig, axes = plt.subplots(figsize=(20, 15), nrows=2, ncols=2, squeeze=True)
fig.suptitle("Grayscale images", fontsize=30)

axes[0][0].imshow(image_1_gray, cmap='gray')
axes[0][0].set_title("airplane", fontsize=15)

axes[0][1].imshow(image_2_gray, cmap='gray')
axes[0][1].set_title("houndog", fontsize=15)

axes[1][0].imshow(image_3_gray, cmap='gray')
axes[1][0].set_title("turkeys", fontsize=15)

axes[1][1].imshow(image_4_gray, cmap='gray')
axes[1][1].set_title("xray", fontsize=15)

# save and display
plt.savefig("images_in_grid.jpg")
plt.show()


