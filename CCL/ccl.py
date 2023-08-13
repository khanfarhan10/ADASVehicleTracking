# python CCL/ccl.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

# fname = r"D:\work\ADASVehicleTracking\visualizations\GMM_image.png" # out of GMM
fname = r"C:\Users\joyde\OneDrive\Desktop\TAMESH DA\aaaa\ADASVehicleTracking-main\visualizations\DSC_1643_GrayScale.png" # out of GMM
# fname = 'D:\work\ADASVehicleTracking\Data\DSC_1643.jpg' # original
def props(img,show_uniques=False):
    print("Shape :",img.shape,"Maximum :",img.max(),"Minimum :",img.min(),"Data Type :",img.dtype)
    if show_uniques:
        print("Uniques :",np.unique(img))
img = cv2.imread(fname, 0)
props(img)
img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
num_labels, labels_im = cv2.connectedComponents(img)

def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    plt.imshow(labeled_img)
    plt.show()
#     cv2.waitKey()

imshow_components(labels_im)
