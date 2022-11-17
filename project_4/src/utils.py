from skimage.io import imread
from skimage.color import rgb2gray

def read_image(path):
    image = imread(path)
    if len(image.shape) > 2:
        return rgb2gray(image)
    return image