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

def get_cluster_images_separated(image, default_background_fill_value = 0, save = True, prepend_save = "GMM_cluster_"):
    uniques = np.unique(image)
    # if default_background_fill_value == "plus_one":
    #     background_fill_value = max(uniques) + 1
    cluster_images = []
    for unique in uniques:
        current_image = np.where(image == unique, 1, default_background_fill_value)
        cluster_images.append(current_image)
        if save:
            save_path = f"visualizations/{prepend_save}{unique}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # modified_image = cv2.cvtColor(current_image, cv2.COLOR_GRAY2RGB)
            # current_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR)
            # if modified_image.shape[2] == 1:
            #     modified_image = cv2.cvtColor(modified_image, cv2.COLOR_GRAY2RGB)
            current_image = current_image / current_image.max()
            current_image = current_image * 255
            cv2.imwrite(save_path, current_image.astype(np.uint8))
        
    return cluster_images
    # print(i)
    