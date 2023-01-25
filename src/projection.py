#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 15:32:43 2021

@author: ceolin
"""

##################################################
## Author: Stefano
## Version: September/October 2021
##################################################

import pandas as pd
import os
from skimage import io, transform
import numpy as np
import copy
from skimage import morphology
from skimage.measure import label, regionprops, block_reduce
from scipy import stats, ndimage
import matplotlib.pyplot as plt
from tifffile import imsave
from tqdm import tqdm
import open3d as o3d
import glob
import shutil


def projections_of_abdomens(
    registered_data_df, registered_folder, destination_folder, landmark_folder, abdomen_mask_filename, n_bins_1 = 180, n_bins_2 = 200):
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
    
    abdomen_mask = io.imread(abdomen_mask_filename)/255
        
    print("Projection of registered 3D stacks to 2D images in progress:")

    registered_data_df[["filename_gfp", "filename_dsred", "filename_tl"]] = registered_data_df.progress_apply(lambda row: \
    project_and_save(row["filename_gfp"], row["filename_dsred"], row["filename_tl"], registered_folder, destination_folder, abdomen_mask, n_bins_1, n_bins_2), axis=1)
    
    registered_data_df["folder"] = destination_folder
    registered_data_df.to_excel(os.path.join(destination_folder,'DatasetInformation.xlsx'))
    
    # Moving the images of channel 1 in the folder for landmarks annotation
    for file in glob.glob(os.path.join(destination_folder,'Projected_C1*.tif')):
        shutil.copy(file,  landmark_folder)
    
    return


def project_and_save(filename_gfp, filename_dsred, filename_tl, folder, destination_folder, abdomen_mask, n_bins_1, n_bins_2):
    
    filename_gfp = os.path.join(folder,filename_gfp)
    filename_dsred = os.path.join(folder,filename_dsred)
    filename_tl = os.path.join(folder,filename_tl)
    
    try:
        Source_gfp =  io.imread(filename_gfp)
        Source_dsred  = io.imread(filename_dsred)
        Source_tl  = io.imread(filename_tl)
    except:
        print("File not found!")
        Source_gfp = float("NaN")
        Source_dsred  = float("NaN")
        Source_tl  = float("NaN")
    
    # Apply mask to select only the fly abdomen:
    Source_gfp = Source_gfp*abdomen_mask
    Source_dsred  = Source_dsred*abdomen_mask
    Source_tl  = Source_tl*abdomen_mask
    
    final_pcd, final_values = image_to_pcd(Source_gfp)
    final_pcd_dsred, final_values_dsred = image_to_pcd(Source_dsred)
    final_pcd_tl, final_values_tl = image_to_pcd(Source_tl)

    n_bins_1=300
    n_bins_2=400
    bins_1 = np.linspace(-210, 210, n_bins_1)
    bins_2 = np.linspace(0, 380, n_bins_2)
    
    projected_image = project_2D(final_pcd, final_values, bins_1, bins_2, stat='max')
    projected_image_dsred = project_2D(final_pcd_dsred, final_values_dsred, bins_1, bins_2, stat='mean')  
    projected_image_tl = project_2D(final_pcd_tl, final_values_tl, bins_1, bins_2, stat='mean')  
    
    projected_image = projected_image.astype(np.uint16)
    projected_image_dsred = projected_image_dsred.astype(np.uint16)
    projected_image_tl = projected_image_tl.astype(np.uint16)

    image_file_names = [os.path.basename(filename_gfp), os.path.basename(filename_dsred), os.path.basename(filename_tl)]
    projected_images = [projected_image, projected_image_dsred, projected_image_tl]
    new_file_names = aux_save_images(projected_images, image_file_names, destination_folder)
    
    return pd.Series([new_file_names[0], new_file_names[1], new_file_names[2]])
    
def aux_save_images(images, names, folder):
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


def project_2D(pcd, values, bins_x1, bins_x2, stat='max'):
    
    indexes = np.asarray(pcd.points).T
    
    x = indexes[2,:]-200
    y = indexes[1,:]-337
    z = indexes[0,:]
    
    x1 = -0.9*y-0.43*x
    y1 = 0.43*y-0.9*x
    
    theta = np.arctan2(y1, z)
    # angle projection:
    ret = stats.binned_statistic_2d(theta, x1, values, statistic=stat, bins=[bins_x1, bins_x2], range=None, expand_binnumbers=False)
    
    # z-projection:
    #ret = stats.binned_statistic_2d(y1, x1, values, statistic=stat, bins=[bins_x1, bins_x2], range=None, expand_binnumbers=False)
    
    final_image = ret.statistic
    
    #df = pd.DataFrame({"x1":x1, "theta":theta, "values":values})
    #df_r = df.round({"x1":1, "theta": 3})
    #df_g = df_r.groupby(by=["x1", "theta"]).max().reset_index()
    #[a,b,c] = df_g.to_numpy().T
    #ret2 = stats.binned_statistic_2d(b, a, c, statistic=stat, bins=[bins_x1, bins_x2], range=None, expand_binnumbers=False)
    #final_image_2 = ret2.statistic
    
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
    
    read_folder = "/media/ceolin/Data/Lab Gompel/Projects/Fly_Abdomens/data_2/03_registered"
    destination_folder = "/media/ceolin/Data/Lab Gompel/Projects/Fly_Abdomens/data_2/04_projected"
    landmark_folder = "../../data_2/05_landmarks/data"
    abdomen_mask_file = "../../data_2/References_and_masks/Reference_abdomen_mask_thick_B_2_2_2.tif"


    df_name = "DatasetInformation.xlsx"
    df = pd.read_excel(os.path.join(read_folder,df_name))
    projections_of_abdomens(df, read_folder, destination_folder, landmark_folder, abdomen_mask_file)

