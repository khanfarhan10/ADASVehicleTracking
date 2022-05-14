import cv2
from utils import readImage, show_image
import numpy as np

def run_morphological_external_transform(image_path = None, img_data = None):
    if image_path is not None:
        img_data = readImage(image_path=image_path,
                             grayscale=True)
    if img_data.max() <= 1:
        img_data = np.uint8(img_data * 255)
    image_shape = img_data.shape
    if len(image_shape) == 3:
        if image_shape[2] != 1:
            img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)
    # props(img_data)
    
    # binarize the image
    binr = cv2.threshold(img_data, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    
    # define the kernel
    kernel = np.ones((3, 3), np.uint8)
    
    # invert the image
    invert = cv2.bitwise_not(binr)

    # use morph gradient
    morph_gradient = cv2.morphologyEx(invert,
                                    cv2.MORPH_GRADIENT,
                                    kernel)
    
    # print the output
    # plt.imshow(morph_gradient, cmap='gray')
    # show_image(morph_gradient, "Morph_gradient", figure_size=(5,5))
    return morph_gradient