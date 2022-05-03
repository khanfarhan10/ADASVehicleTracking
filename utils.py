import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

def props(img,show_uniques=False):
    print("Shape :",img.shape,"Maximum :",img.max(),"Minimum :",img.min(),"Data Type :",img.dtype)
    if show_uniques:
        print("Uniques :",np.unique(img))
        
def show_image(img,title=None):
    plt.figure(figsize=(20,15))
    plt.axis('off')
    plt.imshow(img)
    if title:
        save_path = f"visualizations/{title}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        modified_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # if modified_image.shape[2] == 1:
        #     modified_image = cv2.cvtColor(modified_image, cv2.COLOR_GRAY2RGB)
        modified_image = modified_image / modified_image.max()
        modified_image = modified_image * 255
        cv2.imwrite(save_path, modified_image)
        # cv2.imwrite(save_path, modified_image.astype(np.int8))
        # plt.savefig(save_path, dpi = 600)
    plt.show()
