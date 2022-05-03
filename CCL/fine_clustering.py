import cv2
import numpy as np

def props(img,show_uniques=False):
    print("Shape :",img.shape,"Maximum :",img.max(),"Minimum :",img.min(),"Data Type :",img.dtype)
    if show_uniques:
        print("Uniques :",np.unique(img))

def run_CCL_clustering(image_path = None, img_data = None, convert_BGR2RGB = True,):
    props(img_data)
    if image_path is not None:
        img_data = cv2.imread(image_path, 0)
    if convert_BGR2RGB:
        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
    img_data = img_data.astype(np.uint8) * 255
    img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)
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
