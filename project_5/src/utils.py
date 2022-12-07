import numpy as np
import cv2
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.morphology import (erosion, dilation, opening, area_closing)
from predict import CharacterPredictor

square = np.array([[1,1,1],
                   [1,1,1],
                   [1,1,1]])
                   
def multi_dil(im, num, element=square):
    for i in range(num):
        im = dilation(im, element)
    return im
def multi_ero(im, num, element=square):
    for i in range(num):
        im = erosion(im, element)
    return im

def erosion_dilation(image):
    multi_dilated = multi_dil(image, 7)
    area_closed = area_closing(multi_dilated, 5)
    multi_eroded = multi_ero(area_closed, 11)
    opened_image = opening(multi_eroded)
    return opened_image

def read_image(path):
    image = imread(path)
    if len(image.shape) > 2:
        return rgb2gray(image)
    return image

def bilateral_filter(image, diameter=15, sigma_color=75, sigma_space=75):
    # print(image.max())
    if image.max() != 255:
        image = image * 255
        image = image.astype('uint8')
        return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space) / 255
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)

def median_filter(image, kernel_size=3):
    if image.max() != 255:
        image = image * 255
        image = image.astype('uint8')
        return cv2.medianBlur(image, kernel_size) / 255
    return cv2.medianBlur(image, kernel_size)


def non_local_mean_filter(image, h=3):
    if image.max() != 255:
        image = image * 255
        image = image.astype('uint8')
        return cv2.fastNlMeansDenoising(image, h=h) / 255
    return cv2.fastNlMeansDenoising(image, h=h)

def filter_regions(regions):
    masks = []
    bbox = []
    remove_nums = []
    for num, x in enumerate(regions):
        area = x.area
        convex_area = x.convex_area
        if (convex_area/area > 0.95)  and area>200:
            # print(num, area, convex_area, x.bbox, x.centroid)
            masks.append(regions[num].convex_image)
            bbox.append(regions[num].bbox)   
        else:
            # print(num)
            remove_nums.append(num)
    for i in sorted(remove_nums, reverse=True):
        del regions[i]
    return regions, bbox, masks

def find_space_indexes(regions):
    diff_xs = []
    for index, _ in enumerate(regions):
        if index != 0:
            diff_xs.append((regions[index].bbox[1] - regions[index-1].bbox[3]))
        else:
            diff_xs.append(-1)
    max_diff = sorted(diff_xs)[int(len(diff_xs)*0.90)]
    indexes= [1 if diff_xs[index] >= max_diff else 0 for index, _ in enumerate(regions)]
    # print("index values", diff_xs, indexes, max_diff, len(diff_xs), len(indexes))
    return indexes

def re_arrange_letters(bbox, regions, limit=25):
    bbox = sorted(bbox, key=lambda x: x[0])
    regions = sorted(regions, key=lambda r: r.bbox[0])
    x_mins = list([str(box[0]) for box in bbox])
    # print(x_mins)
    for x_min in x_mins:
        for i in range(1,limit):
            if str(int(x_min)+i) in x_mins:
                # print(str(int(x_min)+i), x_min)
                for k in range(len(x_mins)):
                    if x_mins[k] == str(int(x_min)+i):
                        x_mins[k] = x_min
                # x_mins = list(map(lambda x: x.replace(str(int(x_min)+i), x_min), x_mins))
    # print(x_mins)
    x_mins = [int(x_min) for x_min in x_mins]
    for x_min, box in zip(x_mins, bbox):
        box = (x_min, box[1], box[2], box[3])
    x_mins.append(max(x_mins) + 1)
    unique_x_mins = sorted(list(set(x_mins)))
    # print("Unique:", unique_x_mins)
    line_bbox, line_regions = [], []
    for index, unique_x_min in enumerate(unique_x_mins):
        if index==0:
            start = x_mins.index(unique_x_min)
            continue
        end = x_mins.index(unique_x_min)
        # print(x_mins[start:end])
        line_bbox.append(bbox[start:end])
        line_regions.append(regions[start:end])
        # print(regions[start:end], bbox[start:end])
        start = end
    # print(line_regions)
    new_bbox, new_masks, new_regions, space_indexes, line_end_indexes = [], [], [], [], []
    for line_box, line_region in zip(line_bbox, line_regions):
        # print("Before")
        # [print(r.bbox) for r in line_region]
        line_region = sorted(line_region, key=lambda r: r.bbox[1])
        line_space_indexes = find_space_indexes(regions=line_region)
        space_indexes.extend(line_space_indexes)
        if len(line_end_indexes):
            line_end_indexes.append(line_end_indexes[-1] + len(line_space_indexes))
        else:
            line_end_indexes.append(len(line_space_indexes))
        # print("After")
        # [print(r.bbox) for r in line_region]
        line_box = sorted(line_box, key=lambda x: x[1])
        # print(line_box)
        new_mask = [region.convex_image for region in line_region]
        new_masks.extend(new_mask)
        new_bbox.extend(line_box)
        new_regions.extend(line_region)
    return new_bbox, new_masks, new_regions, space_indexes, line_end_indexes

def predict_letters(image_path, bounded_images, space_indexes, line_end_indexes):
    predictor = CharacterPredictor(model_path="Data/model.pth")
    output_filename = image_path.replace(".jpg", "_predicted.txt")
    output_filename_with_spaces = image_path.replace(".jpg", "_predicted_with_spaces.txt")
    output_file = open(output_filename, "w")
    output_file_with_spaces = open(output_filename_with_spaces, "w")
    for index, bounded_image in enumerate(bounded_images):
        prediction = predictor.predict(bounded_image)
        # print(prediction)
        output_file.write(prediction)
        output_file_with_spaces.write(prediction)
        if index+1 < len(bounded_images):
            if space_indexes[index+1] == 1: 
                output_file_with_spaces.write(" ")
            if index+1 in line_end_indexes:
                output_file_with_spaces.write("\n")
    output_file.close()
    output_file_with_spaces.close()
    return output_filename

def compare_with_gt(gt_filename, predicted_filename):
    # print(gt_filename)
    gt_file = open(gt_filename, "r")
    predicted_file = open(predicted_filename, "r")
    gt_line = gt_file.readline().strip()
    # print(gt_line)
    predicted_line = predicted_file.readline().strip()
    count = 0
    for gt, predicted in zip(gt_line, predicted_line):
        # print(gt, predicted)
        if predicted == gt.upper():
            count+=1
    gt_file.close()
    predicted_file.close()
    return count/len(gt_line)