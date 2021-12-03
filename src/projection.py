#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 15:32:43 2021

@author: ceolin
"""

##################################################
## Function(s) to segment cell nuclei in preprocessed wing images
##################################################
## Author: Stefano
## Version: September/October 2021
##################################################

import pandas as pd
import os
from skimage import io, transform
import numpy as np
import open3d as o3d
import copy
import napari
from skimage import morphology
from skimage.measure import label, regionprops, block_reduce
from scipy import stats, ndimage
import matplotlib.pyplot as plt
from tifffile import imsave
from tqdm import tqdm


def projections_of_abdomens(
        registered_data_df, registered_folder, destination_folder, n_bins_1 = 180, n_bins_2 = 100):
    '''
    Parameters
    ----------
    registered_data_df : pandas dataframe 
        dataframe listing all registered image files 
        and to which construct the images correspond.
    
    registered_folder: str 
        folder were registered images are saved.

    destination_folder: str
        folder were 2D projected images will be saved.

    '''

    tqdm.pandas()
    registered_folder = os.path.join(registered_folder,'')
    
    # clean the destination directory:

    for f in os.listdir(destination_folder):
        os.remove(os.path.join(destination_folder, f))

    print("Projection of registered 3D stacks to 2D images in progress:")

    registered_data_df[["filename_fl", "filename_tl"]] = registered_data_df.progress_apply(lambda row: \
    project_and_save(row["filename_fl"], row["filename_tl"], registered_folder, destination_folder, n_bins_1, n_bins_2), axis=1)
    
    registered_data_df["folder"] = destination_folder
    registered_data_df.to_excel(os.path.join(destination_folder,'DatasetInformation.xlsx'))

    return


def project_and_save(filename_fl, filename_tl, folder, destination_folder, n_bins_1, n_bins_2):
    
    filename_fl = os.path.join(folder,filename_fl)
    filename_tl = os.path.join(folder,filename_tl)
    
    try:
        Source_Image =  io.imread(filename_fl)
        Source_TL  = io.imread(filename_tl)
    except:
        print("File not found!")
        Source_Image = float("NaN")
        Source_TL  = float("NaN")
    
    final_pcd, final_values = image_to_pcd(Source_Image)
    final_pcd_TL, final_values_TL = image_to_pcd(Source_TL)
    
    
    bins_1 = np.linspace(-2,2,n_bins_1)
    bins_2 = np.linspace(0,125,n_bins_2)
    
    projected_image = project_2D(final_pcd, final_values, bins_1, bins_2)
    projected_image_TL = project_2D(final_pcd, final_values_TL, bins_1, bins_2)  


    image_file_names = [os.path.basename(filename_fl), os.path.basename(filename_tl)]
    projected_images = [projected_image, projected_image_TL]
    new_file_names = aux_save_images(projected_images, image_file_names, destination_folder)
    
    return pd.Series([new_file_names[0], new_file_names[1]])
    
def aux_save_images(images,names,folder):
    file_names_list = list()
    if isinstance(images, list):
        for count, image in enumerate(images):
            filename = names[count].replace("Registered_","Projected_" )
            imsave(os.path.join(folder,filename), image)
            file_names_list.append(filename)
    else:
        filename = names
        imsave(os.path.join(folder,filename), image)
        file_names_list.append(filename)
    return list(file_names_list)

def project_2D(pcd, values, bins_x1, bins_x2):
    indexes = np.asarray(pcd.points).T
    x = indexes[1,:]-76
    y = indexes[2,:]-76
    z = indexes[0,:]-55
    
    theta = np.arctan2(y,-z)
    
    ret = stats.binned_statistic_2d(theta, x, values, statistic='max', bins=[bins_x1, bins_x2], range=None, expand_binnumbers=False)
    final_image = ret.statistic
    
    return final_image
    

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


if __name__ == '__main__':
    
    read_folder = "/media/ceolin/Data/Lab Gompel/Projects/Fly_Abdomens/data/03_registered"
    destination_folder = "/media/ceolin/Data/Lab Gompel/Projects/Fly_Abdomens/data/04_projected"

    df_name = "DatasetInformation.xlsx"
    df = pd.read_excel(os.path.join(read_folder,df_name))
    projections_of_abdomens(df, read_folder, destination_folder)

