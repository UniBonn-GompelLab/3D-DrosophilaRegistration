#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function(s) to 2D-project a series of 3D image stacks with multiple separate
channels and save the projected images.

@author: ceolin
"""

import os
import glob
import shutil
import pandas as pd
import numpy as np
from tifffile import imsave
from tqdm import tqdm
from skimage import io
from skimage.filters import gaussian
from scipy import ndimage
from scipy.signal import find_peaks
from scipy.ndimage import convolve1d
from scipy.interpolate import make_interp_spline


def run_2D_projection(
        registered_df, registered_folder, destination_folder, landmark_folder,
        ref_mask_filename, ref_shape_filename, crop_x=None, crop_y=None,
        projection_parameters={'min_y': 0, 'meridian_plane_x': 180,
                               'spline_smoothing': 10, 'projection_radius': 8}):
    '''
    This function loops through all the registered images, calculate the 2D projection,
    crops the final images, saves them in the destination folder and updates or 
    creates the excel file with the information about each image.

    Parameters
    ----------
    registered_df : pandas dataframe
        dataframe listing all registered image files and the corresponding
        genotype and additional info.
    registered_folder : str
        folder were registered images are saved.
    destination_folder : str
        folder were projected images will be saved.
    landmark_folder : str
        folder were to copy projected images for the next step of the analysis.
    ref_mask_filename : str
        path to the 3D binary mask to use in the projection.
    ref_shape_filename : str
        path to the 3D binary shape prior to use in the projection.
    crop_x : int, optional
        if not None, crop the final image to crop_x size along the x axis. 
        The image is cropped around its center.
        The default is None.
    crop_y : int, optional
        if not None, crop the final image to crop_y size along the y axis.
        The image is cropped around its center.
        The default is None.
    projection_parameters: dict, optional
        a dictionary containing the parameters used in the projection functions.        
        
    Returns
    -------
    None.

    '''

    tqdm.pandas()
    registered_folder = os.path.join(registered_folder, '')

    # clean the destination directory:
    for file in os.listdir(destination_folder):
        os.remove(os.path.join(destination_folder, file))

    # open the mask file and the reference shape file and normalize them:
    mask = io.imread(ref_mask_filename)
    mask = mask / np.max(mask)
    shape_ref = io.imread(ref_shape_filename)
    shape_ref = shape_ref / np.max(shape_ref)

    print("Projection of registered 3D stacks to 2D images in progress:")

    columns = ["filename_c1", "filename_c2", "filename_c3"]
    registered_df[columns] = registered_df.progress_apply(
        lambda row: project_and_save_image_stack(
            row["image file name"], registered_folder, destination_folder, mask,
            shape_ref, projection_parameters, crop_x, crop_y),
        axis=1)

    registered_df["folder"] = destination_folder
    registered_df.to_excel(os.path.join(
        destination_folder, 'DatasetInformation.xlsx'), index=False)

    # Moving the images of channel 1 in the folder for landmarks annotation
    for file in glob.glob(os.path.join(destination_folder, 'Projected_C1*.tif')):
        shutil.copy(file,  landmark_folder)

    return


def project_and_save_image_stack(image_file_name, input_folder, destination_folder,
                                 mask, shape_ref, projection_parameters, crop_x=None, crop_y=None):
    filename_c1 = os.path.join(
        input_folder, 'Registered_C1-'+image_file_name)
    filename_c2 = os.path.join(
        input_folder, 'Registered_C2-'+image_file_name)
    filename_c3 = os.path.join(
        input_folder, 'Registered_C3-'+image_file_name)

    try:
        source_c1 = io.imread(filename_c1)
        source_c2 = io.imread(filename_c2)
        source_c3 = io.imread(filename_c3)

    except FileNotFoundError:
        print("File not found: " + image_file_name)
        return pd.Series(["", "", ""])

    # Apply mask to select only the fly abdomen:
    source_c1 = source_c1*mask
    source_c2 = source_c2*mask
    source_c3 = source_c3*mask

    projected_image_c1 = spline_sinusoid_projection_concave_surface(
        source_c2, source_c1, shape_ref, **projection_parameters)
    projected_image_c2 = spline_sinusoid_projection_concave_surface(
        source_c2, source_c2, shape_ref, **projection_parameters)
    projected_image_c3 = spline_sinusoid_projection_concave_surface(
        source_c2, source_c3, shape_ref, **projection_parameters)

    # cropping the projected images around the center:
    if crop_x and crop_y:
        center_y = projected_image_c1.shape[0]/2
        center_x = projected_image_c1.shape[1]/2
        start_y = int(center_y-crop_y/2)
        start_x = int(center_x-crop_x/2)
        projected_image_c1 = projected_image_c1[start_y:(
            start_y+crop_y), start_x:(start_x+crop_x)]
        projected_image_c2 = projected_image_c2[start_y:(
            start_y+crop_y), start_x:(start_x+crop_x)]
        projected_image_c3 = projected_image_c3[start_y:(
            start_y+crop_y), start_x:(start_x+crop_x)]

    projected_image_c1 = projected_image_c1.astype(np.uint16)
    projected_image_c2 = projected_image_c2.astype(np.uint16)
    projected_image_c3 = projected_image_c3.astype(np.uint16)

    # Prepare filenames and save the projected images:
    filename_c1 = 'C1-'+image_file_name
    filename_c2 = 'C2-'+image_file_name
    filename_c3 = 'C3-'+image_file_name

    image_file_names = [filename_c1, filename_c2, filename_c3]
    projected_images = [projected_image_c1,
                        projected_image_c2, projected_image_c3]
    new_file_names = aux_save_images(
        projected_images, image_file_names, "Projected_", destination_folder)

    return pd.Series([new_file_names[0], new_file_names[1], new_file_names[2]])


def aux_save_images(images, names, prefix, folder):
    """
    This function saves a list of images in a folder adding a common prefix to the filenames.

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
            imsave(os.path.join(folder, filename), image)
            file_names_list.append(filename)
    else:
        filename = names
        imsave(os.path.join(folder, filename), image)
        file_names_list.append(filename)

    return list(file_names_list)


def moving_average(array, window_size=3):
    """
    Apply a moving average operation of size window_size on an array.
    The size of the sliding window is reduced at the boundaries down to 1.
    The final array has the same shape of the input array.

    Parameters
    ----------
    array : numpy array
    window_size : integer

    Returns
    -------
    averaged: numpy array
        averaged numpy array

    """
    convolved = convolve1d(array, np.ones(window_size), mode='constant')
    normalization = convolve1d(
        np.ones(len(array)), np.ones(window_size), mode='constant')
    averaged = convolved/normalization
    return averaged


def fit_spline_onto_convex_profile(image, ref_mask, y_min_cutoff=0, smoothing_n=7):
    """
    This function takes an image containing a bright convex curve and a binary mask
    representing prior information about the curve. The function calculates a spline
    interpolation that follows the bright profile in the image and returns two arrays
    containing the x and y coordinates of a set of uniformly distanced points along
    the spline.

    Parameters
    ----------
    image : 2-dim numpy array
    ref_mask  : 2-dim numpy array
    y_min_cutoff: int, optional
        cutoff position of the curve in the image in the y direction.
        The default is 0.
    smoothing_n: int, optional
        Size of the moving average window applied to the curve.
        The default is 7.

    Returns
    -------
    x_new, y_new: numpy arrays
        x and y coordinates of a set of uniformly distributed points along the spline curve.

    """

    ref_dist = ndimage.distance_transform_cdt(ref_mask)
    smooth_image = gaussian(image, 2, preserve_range=False)

    # find starting points of the curve using the reference mask:
    peaks = []
    y_min = 0
    while (len(peaks) < 2) & (y_min < image.shape[0]):

        peaks, _ = find_peaks(ref_dist[y_min, :], distance=10)
        y_min = y_min+1

    if len(peaks) > 1:
        start_x1 = peaks[0]
        start_x2 = peaks[-1]

    else:
        return None

    # use the maxima along y for each position x, as starting points
    # for the spline fitting:
    x_o = np.linspace(start_x1, start_x2, np.abs(start_x1-start_x2), dtype=int)
    y_o = np.argmax(smooth_image[:, x_o], axis=0)
    y_o[y_o < y_min] = y_min

    # reposition first and last point independently from where the brightness peaks are
    y_o[0] = y_min
    y_o[-1] = y_min

    # add an extra point on both ends:
    if y_min_cutoff < y_min:

        y_o = np.insert(y_o, 0, y_min_cutoff)
        y_o = np.append(y_o, y_min_cutoff)

        x_o = np.insert(x_o, 0, x_o[0])
        x_o = np.append(x_o, x_o[-1])

    x_i = moving_average(x_o, window_size=smoothing_n)
    y_i = moving_average(y_o, window_size=smoothing_n)

    y_i[0] = y_min_cutoff
    y_i[-1] = y_min_cutoff

    # resample the points to make the sampling more uniform and
    # create the array containing the distance of each point
    # from the first one along the curve.

    distances = [0]
    x = [x_i[0]]
    y = [y_i[0]]
    dist_from_previous = 0
    minimum_dist = 6

    for i in range(1, len(x_i)):
        dist_from_previous += ((x_i[i]-x_i[i-1]) **
                               2 + (y_i[i]-y_i[i-1])**2)**0.5
        if dist_from_previous > minimum_dist:
            distances.append(distances[-1]+dist_from_previous)
            x.append(x_i[i])
            y.append(y_i[i])
            dist_from_previous = 0

    distances = np.array(distances)
    spline = make_interp_spline(distances, np.c_[x, y], k=2)
    x_new, y_new = spline(np.arange(0, int(max(distances)), 1)).T

    return x_new, y_new


def brightness_along_curve_average(image, pos_x, pos_y, radius=1):
    """
    This function returns the average brightness of an image in areas centered at a set of points.
    It can be used to obtain a brightness profile along an arbitrary curve.

    Parameters
    ----------
    image : 2-dim numpy array
    pos_x : 1-dim numpy array
    pos_y : 1-dim numpy array
    radius : integer, optional
        Distance around each point used for averaging the image brightness.
        The defaults is 1.

    Returns
    -------
    profile: numpy array
        average brightness around the given points.
    """
    profile = []

    for i in range(len(pos_x)):
        r = int(pos_y[i])
        c = int(pos_x[i])
        profile.append(
            np.mean(image[(r-radius):(r+radius), (c-radius):(c+radius)]))

    return np.array(profile)


def brightness_along_curve_perp_max_min(image, x, y, radius=1):
    """
    This function returns the maximum or minimum brightness of an image along an
    arbitrary curve defined by two arrays of x and y coordinates. The brightness
    value at each position x, y is calculated as the maximum along the local
    perpendicular direction to the curve, within a maximum distance from the curve
    defined by the radius parameter.

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

    diff_x = np.concatenate([[0], diff_x_fw])+np.concatenate([diff_x_fw, [0]])
    diff_y = np.concatenate([[0], diff_y_fw])+np.concatenate([diff_y_fw, [0]])

    diff_x[1:-1] = diff_x[1:-1]/2
    diff_y[1:-1] = diff_y[1:-1]/2

    for i in range(len(x)):

        norm = (diff_x[i]**2+diff_y[i]**2)**0.5
        perp_dx = -diff_y[i]/norm
        perp_dy = diff_x[i]/norm

        cols = []
        rows = []
        for j in range(-radius, radius+1):
            cols.append(round(x[i]+perp_dx*j))
            rows.append(round(y[i]+perp_dy*j))
        profile.append(np.max(image[rows, cols]))

    return profile


def spline_sinusoid_projection_concave_surface(image_stack_ref, image_stack_signal,
                                               image_stack_mask, min_y=0, meridian_plane_x=180,
                                               spline_smoothing=10, projection_radius=8):
    """
    This function computes a 2D projection of a 3D image stack with two separate
    channels, using one as a reference to define the surface while using the
    second to read out the brightness values for the projected image.
    The function assumes that the first (ref) image stack represents a bright
    irregular surface, for which each slice in thex-y plane is mostly concave.
    The image_stack_mask is a binary mask that defines an additional prior
    information about the surface shape.
    To calculate the 2D projection the volume is analyzed in slices along its
    last dimension (axis = 2).
    For each slice the profile of the object is interpolated with a spline curve
    and the image brightness is read out along the curve, taking the local maxima
    along the local normal direction. The 1D brightness profile obtained from
    each image slice forms one row of the 2D projected image. The various profiles
    obtained slicing the stack are aligned to each other at the meridian
    plane defined meridian_plane_x.

    Parameters
    ----------
    image_stack_ref : 2-dim numpy array
    image_stack_signal : 2-dim numpy array
    image_stack_mask : 2-dim numpy array
    min_y : int, optional
        Minimium distance of the surface from the beginning of the image stack
        The defaults is 0.
    meridian_plane_x : int, optional
        Position of the meridian plane used for the projection.
        The defaults is 180.
    spline_smoothing = int, optional
        size of the smoothing window applied after spline fitting.
        The defaults is 10.
    projection_radius = int, optional
        defines the maximum distance from the spline curve to consider when looking
        for the maximum brighthness in the perpendicular direction from the spline.
        The default is 8.

    Returns
    -------
    projected: 2-dim numpy array
        The 2d projected image.
    """
    stack_shape = image_stack_ref.shape
    projected = np.zeros([stack_shape[2], stack_shape[1]+4*stack_shape[0]])

    for layer in range(stack_shape[2]):
        image_slice = image_stack_ref[:, :, layer]
        mask_slice = image_stack_mask[:, :, layer]
        profile = fit_spline_onto_convex_profile(
            image_slice, mask_slice, min_y, smoothing_n=spline_smoothing)

        if profile is not None:
            profile_x, profile_y = profile
        else:
            continue

        projected_section = brightness_along_curve_perp_max_min(
            image_stack_signal[:, :, layer], profile_x, profile_y, radius=projection_radius)

        # find center:
        profile_center = np.argmin(np.absolute(profile_x-meridian_plane_x))
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
    df = pd.read_excel(os.path.join(read_folder, df_name))
    run_2D_projection(df, read_folder, destination_folder,
                      landmark_folder, abdomen_mask_file, abdomen_shape_reference_file)
