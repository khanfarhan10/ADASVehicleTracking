# python GMM/coarse_clustering.py -i ../data/GMM/GMM_data.csv -o ../data/GMM/GMM_coarse_clustering.csv -k 3

from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
import cv2

def run_clustering(image_path = None, image = None, k = 3, covariance_type="tied"):
    """Run the GMM Clustering Algorithm on an Image.

    Args:
        image_path (str, optional): _description_. Defaults to None.
        image (_type_, optional): _description_. Defaults to None.
        k (int, optional): _description_. Defaults to 3.
        covariance_type (str, optional): _description_. Defaults to "tied".

    Returns:
        _type_: _description_
    """    
    if image_path is not None:
        image = cv2.imread(image_path)
    img_data = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, (0, 0), fx = 0.5, fy = 0.5)
    # image = image.reshape((image.shape[0] * image.shape[1], 3))
    reshaped_img_data = img_data.reshape(img_data.shape[0]*img_data.shape[1], img_data.shape[2])
    gmm = GaussianMixture(n_components=3, covariance_type="tied")
    
    gmm = GaussianMixture(n_components = k, covariance_type = 'full')
    gmm.fit(image)
    labels = gmm.predict(image)
    labels = labels.reshape((image.shape[0] // 3, 3))
    labels = labels.astype(np.uint8)
    return labels

img_data = cv2.imread(filename = 'D:\work\ADASVehicleTracking\Data\DSC_1643.jpg')
img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)