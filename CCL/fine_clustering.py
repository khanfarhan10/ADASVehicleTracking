import cv2
import numpy as np
import os, sys
root_folder = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0,root_folder)
print(root_folder)

from utils import readImage, props

# https://pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/

def run_CCL_clustering(image_path = None, img_data = None, convert_BGR2RGB = True, connectivity = 4, normalize_labels = True):
    if image_path is not None:
        img_data = readImage(image_path=image_path, grayscale=True, convert_BGR2RGB = convert_BGR2RGB)
    if img_data.max() <= 1:
        img_data = np.uint8(img_data * 255)
    # props(img_data)
    
    thresh_val, thresh_img = cv2.threshold(img_data, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    print(f"thresh_val: {thresh_val}")
    
    # props(thresh_img)
    
    (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(thresh_img, connectivity, cv2.CV_32S)
    
    print(f"Number of labels: {numLabels} found!")
    print(f"Properties of Statistics (stats)")
    props(stats)
    print(f"Properties of centroids:")
    props(centroids)
    print(f"Properties of Labels :")
    props(labels)
    
    if normalize_labels:
        labels = labels/ np.max(labels) * 255
    
    return labels

def run_CCL_clustering_old(image_path = None, img_data = None, convert_BGR2RGB = True,):
    # props(img_data)
    if image_path is not None:
        img_data = readImage(image_path=image_path,
                             grayscale=True, convert_BGR2RGB = convert_BGR2RGB)
    if img_data.max() <= 1:
        img_data = np.uint8(img_data * 255)
    # props(img_data)
    
    # img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)
    img = cv2.threshold(img_data, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
    num_labels, labels_im = cv2.connectedComponents(img)
    # return num_labels, labels_im
    
    label_hue = np.uint8(179*labels_im/np.max(labels_im))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0
    
    return labeled_img
