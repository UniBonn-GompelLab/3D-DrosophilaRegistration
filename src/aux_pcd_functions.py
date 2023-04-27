#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function(s) to transform 3D images in point clouds and viceversa.

@author: ceolin
"""

import numpy as np
import open3d as o3d
from skimage import morphology
from scipy import ndimage


def pcd_to_image(pcd, pcd_values, image_shape):
    """
    Parameters
    ----------
    pcd : open3d pcd.
        an open3d point cloud object.
    pcd_values : 1D numpy array
        brightness of the point cloud points.
    image_shape : list
        shape of the final 3D numpy array.

    Returns
    -------
    image : 3 dimensional numpy array
        grayscale 3D image.

    This function converts a point cloud to a 3D image. If more than one point
    falls in the same voxel their britghness is averaged. Holes in the final
    image are filled with the median values of neighbouring voxels.
    """
    pcd_array = np.asarray(pcd.points).T.astype(int)
    image = np.zeros(image_shape)
    points_count = np.zeros(image_shape)
    assert len(pcd_values) == pcd_array.shape[-1],\
           " the number of points in the pcd object doesn't match the number of britghness values."

    for i in range(pcd_array.shape[-1]):
        pos = tuple(pcd_array[...,i])
        try:
            image[pos] += pcd_values[i]
            points_count[pos] += 1
        except IndexError:
            print("The point cloud does not fit the required image shape.")

    # average the birghtness in voxels containing multiple points:
    image[points_count>1] = image[points_count>1]/points_count[points_count>1]

    # fill holes with the median of neighbouring voxels:
    mask = morphology.closing(image>0, morphology.ball(2))
    image_median = ndimage.median_filter(image, size=3)
    image[points_count==0] = image_median[points_count==0]
    image = image*mask
    return image


def image_to_pcd(image_3d):
    """
    Parameters
    ----------
    image_3d : 3 dimensional numpy array
        the input (sparse) 3d image.

    Returns
    -------
    pcd : open3d pcd.
        an open3d point cloud object.
    pcd_values : 1D numpy array
        brightness of the point cloud points.

    This function converts a 3d image into a point cloud and an array of brightness
    values of each point.

    """
    indexes = np.nonzero(image_3d > 0)
    pcd_points = np.array(indexes).T
    pcd_values = image_3d[indexes]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_points)

    return pcd, pcd_values
