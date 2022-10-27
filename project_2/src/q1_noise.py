from matplotlib import pyplot as plt

from utils import read_image, add_noise

if __name__ == "__main__":
    noise_types = ["gaussian", "replacement", "poisson", "speckle"]
    image_path = "images/xray.png"
    image = read_image(image_path)
    plt.figure(figsize=(16, 16))
    axes = []
    for i, noise in enumerate(noise_types):
        axes.append(plt.subplot(len(noise_types) // 2, 2, i+1))
        try:
            noisy_image = add_noise(image.copy(), noise)
            title = f"Noisy image ({noise})"
        except NotImplementedError:
            noisy_image = image
            title = f"Noisy image ({noise}-Error)"
        axes[i].imshow(noisy_image, cmap="gray")
        axes[i].set_title(title, fontsize=15)
    output_path = image_path.replace(image_path.split("/")[-1], "output_noise_"+image_path.split("/")[-1])
    print("plotting image: ", output_path)
    plt.savefig(output_path)
    plt.show()
