##################################################
## Function(s) to transform images in point clouds 
## and viceversa
##################################################
## Author: Stefano
##################################################

import numpy as np
import open3d as o3d
import copy
from skimage import morphology
from scipy import ndimage


def pcd_to_image(pcd,values, image_shape):
    array = np.asarray(pcd.points).T.astype(int)
    image = np.zeros(image_shape)
    count = np.zeros(image_shape)
    for i in range(array.shape[-1]):
        pos = tuple(array[...,i])
        if (pos[0]<image_shape[0]) and (pos[1]<image_shape[1]) and (pos[2]<image_shape[2]) and (pos[0]>0) and (pos[1]>0) and (pos[2]>0):
            image[pos] += values[i]
            count[pos] += 1
    image[count>1] = image[count>1]/count[count>1]
    mask = morphology.closing(image>0, morphology.ball(2))
    image_median = ndimage.median_filter(image, size=3)
    image[count==0] = image_median[count==0]
    image = image*mask
    return image

def image_to_pcd(image):
    indexes = np.nonzero(image > 0)
    pcd_points = np.array(indexes).T
    pcd_values = image[indexes]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_points)
    return pcd, pcd_values