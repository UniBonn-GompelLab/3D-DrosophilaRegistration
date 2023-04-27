#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function(s) to calculate a 2d projection of the surface of a registered 
3d fluorescence image of a fly abdomen.

@author: ceolin
"""

import pandas as pd
import os
from skimage import io
import numpy as np
from scipy import stats
from tifffile import imsave
from tqdm import tqdm
import glob
import shutil
from scipy.signal import find_peaks
from scipy.ndimage import convolve1d
from skimage.filters import gaussian
from scipy.interpolate import make_interp_spline
from scipy import ndimage

if __name__ == '__main__':
    from aux_pcd_functions  import pcd_to_image, image_to_pcd
else:
    from src.aux_pcd_functions  import pcd_to_image, image_to_pcd


def projections_of_abdomens(
    registered_data_df, registered_folder, destination_folder, landmark_folder, abdomen_mask_filename, abdomen_shape_reference_file , crop_x = None, crop_y = None):
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
    abdomen_shape_ref = io.imread(abdomen_shape_reference_file)/255
    
        
    print("Projection of registered 3D stacks to 2D images in progress:")

    registered_data_df[["filename_gfp", "filename_dsred", "filename_tl"]] = registered_data_df.progress_apply(lambda row: \
    project_and_save(row["image file name"], row["filename_gfp"], row["filename_dsred"], row["filename_tl"], registered_folder, destination_folder, abdomen_mask, abdomen_shape_ref , crop_x, crop_y), axis=1)
    
    registered_data_df["folder"] = destination_folder
    registered_data_df.to_excel(os.path.join(destination_folder,'DatasetInformation.xlsx'))
    
    # Moving the images of channel 1 in the folder for landmarks annotation
    for file in glob.glob(os.path.join(destination_folder,'Projected_C1*.tif')):
        shutil.copy(file,  landmark_folder)
    
    return


def project_and_save(image_file_name, filename_gfp, filename_dsred, filename_tl, folder, destination_folder, abdomen_mask, abdomen_shape_ref,  crop_x = None, crop_y = None):
    """

    Parameters
    ----------
    image_file_name : TYPE
        DESCRIPTION.
    filename_gfp : TYPE
        DESCRIPTION.
    filename_dsred : TYPE
        DESCRIPTION.
    filename_tl : TYPE
        DESCRIPTION.
    folder : TYPE
        DESCRIPTION.
    destination_folder : TYPE
        DESCRIPTION.
    abdomen_mask : TYPE
        DESCRIPTION.
    abdomen_shape_ref : TYPE
            DESCRIPTION.
    crop_x : TYPE, optional
        DESCRIPTION. The default is None.
    crop_y : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
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
    Source_dsred = Source_dsred*abdomen_mask
    Source_tl = Source_tl*abdomen_mask
        
    projected_image_gfp = fly_abdomen_spline_projection(Source_dsred, Source_gfp, abdomen_shape_ref)
    projected_image_dsred = fly_abdomen_spline_projection(Source_dsred, Source_dsred, abdomen_shape_ref)
    projected_image_tl = fly_abdomen_spline_projection(Source_dsred, Source_tl, abdomen_shape_ref, maxima = False)
    
    # cropping the projected images around the center:
    if crop_x and crop_y:
        center_y = projected_image_gfp.shape[0]/2
        center_x = projected_image_gfp.shape[1]/2
        start_y = int(center_y-crop_y/2)
        start_x = int(center_x-crop_x/2)
        projected_image_gfp = projected_image_gfp[start_y:(start_y+crop_y),start_x:(start_x+crop_x)]
        projected_image_dsred = projected_image_dsred[start_y:(start_y+crop_y),start_x:(start_x+crop_x)]
        projected_image_tl = projected_image_tl[start_y:(start_y+crop_y),start_x:(start_x+crop_x)]        
        
    projected_image_gfp = projected_image_gfp.astype(np.uint16)
    projected_image_dsred = projected_image_dsred.astype(np.uint16)
    projected_image_tl = projected_image_tl.astype(np.uint16)
    
    # Prepare filenames and save the projected images:
    filename_GFP = os.path.join(folder,'C1-'+image_file_name)
    filename_DsRed = os.path.join(folder,'C2-'+image_file_name)
    filename_TL = os.path.join(folder,'C3-'+image_file_name)

    image_file_names = [os.path.basename(filename_GFP), os.path.basename(filename_DsRed), os.path.basename(filename_TL)]
    projected_images = [projected_image_gfp, projected_image_dsred, projected_image_tl]
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


def moving_average(x, n = 3):
    """
    Apply a moving average operation of size n on the array x. 
    The size of the sliding window is reduced at the boundaries of x down to 1.
    The final array has the same shape of x.
    
    Parameters
    ----------
    x : numpy array
    n : integer
        
    Returns
    -------
    averaged: numpy array
        averaged numpy array

    """
    convolved = convolve1d(x, np.ones(n), mode='constant')
    normalization = convolve1d(np.ones(len(x)), np.ones(n), mode = 'constant')
    averaged = convolved/normalization
    return averaged

def fit_spline_onto_convex_profile(image, ref_mask, y_min_cutoff = 20, smoothing_n = 7):
    """
    This function takes an image containing a bright convex curve and a binary mask representing
    prior information about the curve. The function calculates a spline interpolation that follows the
    bright profile in the image and returns two arrays containing the x and y coordinates of 
    a set of uniformly distanced points along the spline.
    
    Parameters
    ----------
    image : 2-dim numpy array
    ref_mask  : 2-dim numpy array
    y_min_cutoff: int, optional
        cutoff position of the curve in the image in the y direction.
        The default is 20.
    smoothing_n: int, optional
        Size of the moving average window applied to the curve.
        The default is 7.
        
    Returns
    -------
    x_new, y_new: numpy arrays
        x and y coordinates of a set of uniformly distributed points along the spline curve.

    """
    
    ref_dist    = ndimage.distance_transform_cdt(ref_mask)
    image_dist   = ndimage.distance_transform_cdt(image > 0)
    smooth_image = gaussian(image, 2, preserve_range=False)
    
    # find starting points of the curve using the reference mask:
    peaks = []
    y_min = 0
    while (len(peaks) < 2) & (y_min < image.shape[0]):
        
        peaks, _ = find_peaks(ref_dist[y_min,:], distance = 10)
        y_min    = y_min+1
    
    if len(peaks) > 1:
        start_x1 = peaks[0]
        start_x2 = peaks[-1]
    
    else:
        return None
    
    # use the maxima along y for each position x, as starting points
    # for the spline fitting:
    x_o = np.linspace(start_x1, start_x2, np.abs(start_x1-start_x2), dtype = int)
    y_o = np.argmax(smooth_image[:,x_o], axis = 0)
    y_o[y_o < y_min] = y_min

    # reposition first and last point independently from where the brightness peaks are
    y_o[0]  = y_min
    y_o[-1] = y_min

    # add an extra point on both ends:
    if y_min_cutoff < y_min:
        
        y_o = np.insert(y_o, 0, y_min_cutoff)
        y_o = np.append(y_o, y_min_cutoff)

        x_o = np.insert(x_o, 0, x_o[0])
        x_o = np.append(x_o, x_o[-1])

    x_i = moving_average(x_o, n = smoothing_n)
    y_i = moving_average(y_o, n = smoothing_n)

    y_i[0]  = y_min_cutoff
    y_i[-1] = y_min_cutoff


    # resample the points to make the sampling more uniform and
    # create the array containing the distance of each point
    # from the first one along the curve.

    distances = [0]
    x = [x_i[0]]
    y = [y_i[0]]
    previous = 0
    dist_from_previous = 0
    minimum_dist = 6

    for i in range(1, len(x_i)):
        dist_from_previous += ((x_i[i]-x_i[i-1])**2 + (y_i[i]-y_i[i-1])**2)**0.5
        if dist_from_previous > minimum_dist:
            distances.append(distances[-1]+dist_from_previous)
            x.append(x_i[i])
            y.append(y_i[i])
            dist_from_previous = 0
            
    distances = np.array(distances)
    spline = make_interp_spline(distances, np.c_[x, y], k=2)
    x_new, y_new = spline(np.arange(0,int(max(distances)),1)).T
    
    return x_new, y_new

def brightness_along_curve_average(image, x, y, radius = 1):
    """
    This function returns the average brightness of an image in areas centered at a set of points.
    It can be used to obtain a brightness profile along an arbitrary curve.
    
    Parameters
    ----------
    image : 2-dim numpy array
    x : 1-dim numpy array
    y : 1-dim numpy array
    radius : integer, optional
        Distance around each point used for averaging the image brightness.
        The defaults is 1.
        
    Returns
    -------
    profile: numpy array
        average brightness around the given points.
    """
    profile = []

    for i in range(len(x)):
        r = int(y[i])
        c = int(x[i])
        profile.append(np.mean(image[(r-radius):(r+radius),(c-radius):(c+radius)]))
    
    return np.array(profile)


def brightness_along_curve_perp_max_min(image, x, y, radius = 1, maxima = True):
    """
    This function returns the maximum or minimum brightness of an image along an arbitrary curve
    defined by two arrays of x and y coordinates. The brightness value at each position x, y
    is calculated as the maximum along the local perpendicular direction to the curve, within a maximum 
    distance from the curve defined by the radius parameter.
    
    Parameters
    ----------
    image : 2-dim numpy array
    x : 1-dim numpy array
    y : 1-dim numpy array
    radius : integer, optional
        Maximum distance from each point to be considered when calculating the maximum brightness.
        The defaults is 1.
        
    Returns
    -------
    profile: numpy array
        maximum projected brightness along the curve.
    """

    profile = []
    
    diff_x_fw = x[1:]-x[:-1]
    diff_y_fw = y[1:]-y[:-1]
    
    diff_x = np.concatenate([[0], diff_x_fw])+np.concatenate([diff_x_fw,[0]])
    diff_y = np.concatenate([[0], diff_y_fw])+np.concatenate([diff_y_fw,[0]])
    
    diff_x[1:-1] = diff_x[1:-1]/2 
    diff_y[1:-1] = diff_y[1:-1]/2 
    
    for i in range(len(x)):
        
        norm = (diff_x[i]**2+diff_y[i]**2)**0.5
        perp_dx = -diff_y[i]/norm
        perp_dy = diff_x[i]/norm
        
        cols = []
        rows = []
        for j in range(-radius, radius+1):
            cols.append( round(x[i]+perp_dx*j) )
            rows.append( round(y[i]+perp_dy*j) )
        if maxima:
            profile.append(np.max(image[rows, cols]))
        else:
            profile.append(np.min(image[rows, cols]))
            
    return profile 


def fly_abdomen_spline_projection(image_stack_ref, image_stack_signal, image_stack_mask, min_y = 0, center = 180, maxima = True):
    """
    This function performs the 2d projection of a 3d image stack of a fly abdomen 
    with two fluorescent channels. The first channel is the reference channel and
    is used to define the abdomen surface. The second channel is the signal of interest.
    To calculate the 2d projection the abdomen is analyzed in slices along its axis.
    For each slice the profile of the abdomen is interpolated with a spline curve 
    and the image brightness is read out along the curve, taking the local maxima along
    the perpendicular. The 1d brightness profile obtained from each image slice forms
    one row of the 2d projected image.
    
    Parameters
    ----------
    image_stack_ref : 2-dim numpy array
    image_stack_signal : 2-dim numpy array
    image_stack_mask : 2-dim numpy array
    min_y : integer, optional
        Minimium distance of the abdomen from the beginning of the image stack
        The defaults is 0.
    center_x : integer, optional
        center of the abomen in the x direction. Used to align the profiles obtained from different slices.
        The defaults is 180.
        
    Returns
    -------
    projected: 2-dim numpy array
        The 2d projected image.
    """  
    stack_shape = image_stack_ref.shape
    projected = np.zeros([stack_shape[2], stack_shape[1]+4*stack_shape[0]])
    
    for layer in range(stack_shape[2]):
        image_slice  = image_stack_ref[:,:,layer]
        mask_slice   = image_stack_mask[:,:,layer]
        profile = fit_spline_onto_convex_profile(image_slice, mask_slice, min_y, smoothing_n = 10)

        if profile is not None:
            profile_x, profile_y = profile
        else:
            continue
        
        projected_section = brightness_along_curve_perp_max_min(image_stack_signal[:,:, layer], profile_x, profile_y, radius = 8, maxima = maxima)

        # find center:
        profile_center = np.argmin(np.absolute(profile_x-center))
        projection_min = round(projected.shape[1]/2)-profile_center
        projection_max = projection_min+len(projected_section)
        projected[-layer, projection_min:projection_max] = projected_section
        
    return projected

if __name__ == '__main__':
    
    read_folder = "../../data/03_registered"
    destination_folder = "../../data/04_projected"
    landmark_folder = "../../data/05_landmarks/data"
    abdomen_mask_file = "../../data/References_and_masks/Reference_abdomen_mask_iso_thick.tif"
    abdomen_shape_reference_file = "../../data/References_and_masks/Reference_abdomen_mask_iso.tif"



    df_name = "DatasetInformation.xlsx"
    df = pd.read_excel(os.path.join(read_folder,df_name))
    projections_of_abdomens(df, read_folder, destination_folder, landmark_folder, abdomen_mask_file, abdomen_shape_reference_file )

