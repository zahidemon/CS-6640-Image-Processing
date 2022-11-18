from matplotlib import pyplot as plt

from utils import load_images_from_txt, create_mosaic

if __name__ == "__main__":
    text_file_name = "read.txt"
    directory = "data/"
    image_dict = load_images_from_txt(directory, text_file_name)

    cells_mosaic = create_mosaic(image_dict)
    plt.figure(figsize=(20, 20))
    plt.title('Image mosaic of cells')
    plt.imshow(cells_mosaic, cmap='gray')

    output_filepath = "q2_mosaic.jpg"
    plt.savefig(output_filepath)
    plt.show()

