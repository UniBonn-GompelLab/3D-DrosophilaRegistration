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
from skimage import io
import numpy as np
from scipy import stats
from tifffile import imsave
from tqdm import tqdm
import glob
import shutil

if __name__ == '__main__':
    from aux_pcd_functions  import pcd_to_image, image_to_pcd
else:
    from src.aux_pcd_functions  import pcd_to_image, image_to_pcd


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
    project_and_save(row["image_file_name"], row["filename_gfp"], row["filename_dsred"], row["filename_tl"], registered_folder, destination_folder, abdomen_mask, n_bins_1, n_bins_2), axis=1)
    
    registered_data_df["folder"] = destination_folder
    registered_data_df.to_excel(os.path.join(destination_folder,'DatasetInformation.xlsx'))
    
    # Moving the images of channel 1 in the folder for landmarks annotation
    for file in glob.glob(os.path.join(destination_folder,'Projected_C1*.tif')):
        shutil.copy(file,  landmark_folder)
    
    return


def project_and_save(image_file_name, filename_gfp, filename_dsred, filename_tl, folder, destination_folder, abdomen_mask, n_bins_1 = 300, n_bins_2 = 400):
    
    try:
        Source_gfp   = io.imread( os.path.join(folder,filename_gfp) )
        Source_dsred = io.imread( os.path.join(folder,filename_dsred) )
        Source_tl    = io.imread( os.path.join(folder,filename_tl) )
    except:
        print(" ****** File not found! *********")
        print("Can not open the file: "+image_file_name)
        return pd.Series(["","",""])
    
    # Apply mask to select only the fly abdomen:
    Source_gfp = Source_gfp*abdomen_mask
    Source_dsred  = Source_dsred*abdomen_mask
    Source_tl  = Source_tl*abdomen_mask
    
    final_pcd, final_values = image_to_pcd(Source_gfp)
    final_pcd_dsred, final_values_dsred = image_to_pcd(Source_dsred)
    final_pcd_tl, final_values_tl = image_to_pcd(Source_tl)

    bins_1 = np.linspace(-210, 210, n_bins_1)
    bins_2 = np.linspace(0, 380, n_bins_2)
    
    projected_image = project_2D_cylindrical_xy(final_pcd, final_values, bins_1, bins_2, stat='max')
    projected_image_dsred = project_2D_cylindrical_xy(final_pcd_dsred, final_values_dsred, bins_1, bins_2, stat='mean')  
    projected_image_tl = project_2D_cylindrical_xy(final_pcd_tl, final_values_tl, bins_1, bins_2, stat='mean')  

    #projected_images = project_2D_new(Source_gfp, Source_dsred, Source_tl)
    #[projected_image, projected_image_dsred, projected_image_tl] = projected_images
    
    projected_image = projected_image.astype(np.uint16)
    projected_image_dsred = projected_image_dsred.astype(np.uint16)
    projected_image_tl = projected_image_tl.astype(np.uint16)
    
    # Prepare filenames and save the projected images:
    filename_GFP = os.path.join(folder,'C1-'+image_file_name)
    filename_DsRed = os.path.join(folder,'C2-'+image_file_name)
    filename_TL = os.path.join(folder,'C3-'+image_file_name)

    image_file_names = [os.path.basename(filename_GFP), os.path.basename(filename_DsRed), os.path.basename(filename_TL)]
    projected_images = [projected_image, projected_image_dsred, projected_image_tl]
    new_file_names = aux_save_images(projected_images, image_file_names, "Projected_", destination_folder)
    
    return pd.Series([new_file_names[0], new_file_names[1], new_file_names[2]])
    
def aux_save_images(images, names, prefix, folder):
    """
    Save a list of images in a folder adding a common prefix to the filenames.

    Parameters
    ----------
    images : numpy array or list of numpy arrays
        image(s) to save.
    names : string or list of strings
        root name(s) of the images.
    prefix : string
        prefix to add to the filename.
    folder : string
        destination folder.

    Returns
    -------
    list
        list of new file names.

    """
    file_names_list = list()
    
    if isinstance(images, list):
        for count, image in enumerate(images):
            
            filename = prefix+names[count]
            imsave(os.path.join(folder,filename), image)
            file_names_list.append(filename)
    else:
        filename = names
        imsave(os.path.join(folder,filename), image)
        file_names_list.append(filename)
    
    return list(file_names_list)


def project_2D_cylindrical_xy(pcd, values, bins_theta, bins_z,  x0=200, y0=337, cos_theta=-0.9, sin_theta=0.43, stat='max'):
    """
    Function that computes the 2D projection of the pre aligned point cloud 
    object describing the abdomen surface. 
    The 2D projection is computed in cylindrical coordinates with the cylinder 
    axis laying on the xy plane.
    
    Parameters
    ----------
    pcd : point cloud
    values : brightness values of pcd
    bins_theta : numpy array
        limits of bins for angle.
    bins_z : numpy array
        limits of bins along cylinder axis.
    x0 : float, optional
        x origin. The default is 200.
    y0 : float, optional
        y origin. The default is 337.
    cos_theta : float
        x coordinate of orientation of cylinder axis.
    sin_theta : float
        y coordinate of orientation of cylinder axis.
    stat : string, optional
        statistic to use when binning values from the pcd. The default is 'max'.

    Returns
    -------
    final_image : numpy array
        projected 2d image.

    """
    
    indexes = np.asarray(pcd.points).T
    
    x = indexes[2,:]-x0
    y = indexes[1,:]-y0
    z = indexes[0,:]
    
    parallel = cos_theta*y-sin_theta*x
    perp = sin_theta*y-cos_theta*x
    
    theta = np.arctan2(perp, z)
    
    # angle projection:
    ret = stats.binned_statistic_2d(theta, parallel, values, statistic=stat, bins=[bins_theta, bins_z], range=None, expand_binnumbers=False)
    
    # z-projection:
    #ret = stats.binned_statistic_2d(y1, x1, values, statistic=stat, bins=[bins_x1, bins_x2], range=None, expand_binnumbers=False)
    
    final_image = ret.statistic
    
    return final_image

if __name__ == '__main__':
    
    read_folder = "../../data_2/03_registered"
    destination_folder = "../../data_2/04_projected"
    landmark_folder = "../../data_2/05_landmarks/data"
    abdomen_mask_file = "../../data_2/References_and_masks/Reference_abdomen_mask_iso.tif"


    df_name = "DatasetInformation.xlsx"
    df = pd.read_excel(os.path.join(read_folder,df_name))
    projections_of_abdomens(df, read_folder, destination_folder, landmark_folder, abdomen_mask_file)

