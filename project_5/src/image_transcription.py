from matplotlib import pyplot as plt
import numpy as np
from skimage.transform import resize
from skimage.measure import label, regionprops
from utils import bilateral_filter, compare_with_gt, filter_regions, predict_letters, re_arrange_letters, read_image

if __name__ == "__main__":
    file_names = [
        "Data/test_images/noisy_one_paragraph.jpg",
        "Data/test_images/noisy_three_sentences.jpg"
    ]

    gt_filenames = {
        "Data/test_images/noisy_one_paragraph.jpg": "Data/test_images/one_paragraph_gt.txt",
        "Data/test_images/noisy_three_sentences.jpg": "Data/test_images/three_sentences_gt.txt" 
    }

    threshold = 0.75

    for image_path in file_names:
        print("Filename: ", image_path)
        image = read_image(image_path) 
        filtered_image = bilateral_filter(image)
        binarized_image = filtered_image < threshold
        label_im = label(binarized_image)
        regions = regionprops(label_im)
        print("Total number of regions: ", len(regions))
        regions, bbox, masks = filter_regions(regions)
        print("Number of components after filtering: ", len(regions))
        count = len(masks)
        fig, ax = plt.subplots(15, int(count/15), figsize=(60,60))
        bounded_images = []
        bbox, masks, regions, space_indexes, line_end_indexes = re_arrange_letters(bbox, regions)
        # print(len(masks), line_end_indexes)
        for index, (axis, box, mask) in enumerate(zip(ax.flatten(), bbox, masks)):
            bounded_image  =  binarized_image[box[0]:box[2], box[1]:box[3]] * mask
            bounded_image = np.pad(bounded_image, pad_width=10)
            bounded_image = bilateral_filter(bounded_image, 1)
            bounded_image = resize(bounded_image, (28, 28))
            bounded_images.append(bounded_image)
            axis.imshow(bounded_image, cmap="gray")
        fig.tight_layout()
        fig.savefig(image_path.replace(".jpg", "_letters.jpg"))

        output_filename = predict_letters(image_path, bounded_images, space_indexes, line_end_indexes)

        accuracy = compare_with_gt(
            gt_filename=gt_filenames[image_path], 
            predicted_filename= output_filename
            )
        
        print("Accuracy: ", accuracy)
        print()