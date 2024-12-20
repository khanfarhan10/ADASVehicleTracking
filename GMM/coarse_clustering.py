# python GMM/coarse_clustering.py -i ../data/GMM/GMM_data.csv -o ../data/GMM/GMM_coarse_clustering.csv -k 3

from sklearn.mixture import GaussianMixture
import numpy as np
import cv2
from utils import readImage


def run_GMM_clustering(image_path = None, img_data = None, convert_BGR2RGB = True,  k = 3, covariance_type="tied"):  
    """Run the GMM Clustering Algorithm on an Image.

    Args:
        image_path (str, optional): Path to the image. Defaults to None.
        img_data (np.array, optional): Array of the Image read by Numpy/CV2. Defaults to None.
        convert_BGR2RGB (bool, optional): Convert the BGR image to RGB. Defaults to True.
        k (int, optional): The number of clusters for which clustering will be performed. Defaults to 3.
        covariance_type (str, optional): Parameter for GMM from sklearn. Defaults to "tied", "full".
        # TODO :  Add GMM_vars = {} to the input!

    Returns:
        np.array: Clustered GMM Image
    """
    if image_path:
        img_data = readImage(image_path=image_path, convert_BGR2RGB=convert_BGR2RGB)
    if k == "elbow_method":
        k = 4 # TODO - Implement the elbow method

    reshaped_img_data = img_data.reshape(img_data.shape[0]*img_data.shape[1], img_data.shape[2])
    gmm = GaussianMixture(n_components = k, covariance_type = covariance_type)
    gmm = gmm.fit(reshaped_img_data)
    
    cluster = gmm.predict(reshaped_img_data)
    retransformed_clustered_image = cluster.reshape(img_data.shape[0], img_data.shape[1], 1).astype(np.uint8)
    # plt.imshow(retransformed_clustered_image)
    return retransformed_clustered_image
