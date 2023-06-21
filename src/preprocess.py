#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function(s) to segment a series of 3D image stacks with multiple separate
channels and save the prperocessed images.

@author: ceolin
"""

import os
import pandas as pd
from skimage import io, transform
import numpy as np
import open3d as o3d
from skimage import morphology
from skimage.measure import label, regionprops
from scipy.signal import find_peaks
from tifffile import imsave
from tqdm import tqdm

if __name__ == '__main__':
    from aux_pcd_functions import pcd_to_image, image_to_pcd
else:
    from src.aux_pcd_functions import pcd_to_image, image_to_pcd


def preprocess_and_segment_images(
        read_folder, destination_folder, downscaling=(1, 1, 1), bit_depth=8, only_on_new_files=True,
        segmentation_parameters={'threshold': 1.05, 'max_iter': 200, 'fraction_range': [
            0.04, 0.05], 'padding': 20, 'closing_r': 4, 'dilation_r': 8, 'mesh_radius': 30},
        database_filename='DatasetInformation.xlsx'):
    '''
    Parameters
    ----------
    read_folder : str
        path of the folder containing the raw data.
    destination_folder : str
        path of the folder where preprocessed images will be saved.
    downscaling: (float, float, float), optional
        downscaling factor along z, x, y, in pixels.
    bit_depth: int, optional
        bit depth of raw data
    only_on_new_files: bool, optional
        whether to reanalyze the whole dataset or only new images.
        The default is True
    segmentation_parameters: dict, optional
        dictionary specifing the values of the parameters used for segmentation.
    database_filename : str, optional
        name of the excel file(s) included with the raw data.
        The default is 'DatasetInformation.xlsx'.

    Returns
    -------
    None.

    The function collects raw images across all subfolders in the provided folder.
    It removes duplicated images, downscale and pad the images, run a segmentation
    algorithm that selects only the surface of the fly in the original z-stack and
    save the final 3d stacks in the destination folder.

    NOTE: Channel 1 is used as a reference for segmentation. The preprocessed
    images are rescaled to 16 bits.

    '''

    tqdm.pandas()

    # unify databases across folders:
    raw_data_df = create_raw_images_database(read_folder, database_filename)

    if len(raw_data_df) == 0:
        print(database_filename +
              ': file not found in the input folder or its subfolders.')
        return

    # In case we want to reanalyze the entire dataset clean the destination directory:
    if not only_on_new_files:
        for file in os.listdir(destination_folder):
            os.remove(os.path.join(destination_folder, file))

    # Look for the database file of the preprocessed images to check which
    # images have been already processed:
    try:
        preproc_df = pd.read_excel(os.path.join(
            destination_folder, database_filename))
    # if the file doesn't exist create an empty dataframe
    except FileNotFoundError:
        preproc_df = pd.DataFrame(
            columns=["experiment", "filename_c1", "filename_c2", "filename_c3"])

    # run row by row over database of files, downsample, segment and save the images,
    # return filenames to create new DatasetInformation.xlsx file in the destination folder
    print("Preprocessing of raw images in progress:")
    new_columns = ["experiment", "filename_c1", "filename_c2", "filename_c3"]
    raw_data_df[new_columns] = raw_data_df.progress_apply(lambda row:
                            preprocess_and_save(
                                row["image file name"], row["folder"],
                                downscaling, bit_depth, destination_folder,
                                preproc_df, segmentation_parameters)
                            , axis=1)

    # remove rows with NaNs and save the dataframe as an excel file
    raw_data_df = raw_data_df[raw_data_df['experiment'].notna()]
    raw_data_df.to_excel(os.path.join(
        destination_folder, 'DatasetInformation.xlsx'), index=False)

    return


def create_raw_images_database(raw_data_folder, database_filename='DatasetInformation.xlsx'):
    """
    Parameters
    ----------
    raw_data_folder : str
        path of the folder containing the raw data.
    database_filename : str, optional
        name of the excel file(s) included with the raw data.
        The default is 'DatasetInformation.xlsx'.

    Returns
    -------
    raw_data_df : pandas dataframe
        dataframe containing the filenames of all the input raw images in all
        subfolders, after removing duplicated files.

    """
    raw_data_df = pd.DataFrame()
    for directory, _, _ in os.walk(raw_data_folder):
        filename = os.path.join(directory, database_filename)
        try:
            dataset_info = pd.read_excel(filename)
            dataset_info['folder'] = str(os.path.join(directory, ''))
            raw_data_df = raw_data_df.append(dataset_info)
        except FileNotFoundError:
            pass

    raw_data_df['File_exists'] = raw_data_df.apply(lambda row:
                                                   pd.Series(os.path.isfile(os.path.join(row['folder'], 'C1-'+row['image file name']))), axis=1)

    raw_data_df = raw_data_df[raw_data_df['File_exists']]
    raw_data_df = raw_data_df.drop_duplicates(
        subset='image file name', keep="last")

    return raw_data_df


def preprocess_and_save(image_file_name, folder, downscaling, bit_depth, destination_folder, preproc_df, segm_pars):
    """
    This function downscale and pad the three channels of one image stack,
    runs a segmentation algorithm on the first channel to selects only the
    surface of the fly save the final 3D stacks in the destination folder.

    NOTE: Channel 1 is used as a reference for segmentation. The preprocessed
    images are rescaled to 16 bits.

    Parameters
    ----------
    image_file_name : str
        name of the image stack to preprocess.
    folder : str
        name of the input data folder.
    downscaling: (float, float, float)
        downscaling factor along z, x, y, in pixels.
    bit_depth: int
        bit depth of raw data.
    destination_folder : str
        path of the folder where preprocessed images will be saved.
    preproc_df : pandas dataframe
        dataframe containing the filename of the preprocessed images, used to
        check if an image should be skipped in the preprocessing.

    Returns
    -------
    pandas series
        return the name of the original image stack and the names of the three
        preprocessed channels.

    """
    filename_c1 = os.path.join(folder, 'C1-'+image_file_name)
    filename_c2 = os.path.join(folder, 'C2-'+image_file_name)
    filename_c3 = os.path.join(folder, 'C3-'+image_file_name)

    if os.path.splitext(image_file_name)[0] in preproc_df['experiment'].values:
        return pd.Series([os.path.splitext(image_file_name)[0], 'Preprocessed_C1-'+image_file_name, 'Preprocessed_C2-'+image_file_name, 'Preprocessed_C3-'+image_file_name])

    try:
        image_c1 = io.imread(filename_c1)
        image_c2 = io.imread(filename_c2)
        image_c3 = io.imread(filename_c3)

    except FileNotFoundError:
        print("File not found: " + image_file_name)
        image_c1 = float("NaN")
        image_c2 = float("NaN")
        image_c3 = float("NaN")
        return pd.Series([float("NaN"), float("NaN"), float("NaN"), float("NaN")])

    # rescale images to 16bits:
    max_value = 2**bit_depth-1

    image_c1 = image_c1*65536/max_value
    image_c2 = image_c2*65536/max_value
    image_c3 = image_c3*65536/max_value

    # Resizing of the images:
    new_image_shape = [int(image_c1.shape[i]/downscaling[i]) for i in range(3)]
    image_c1_downscaled = transform.resize(
        image_c1, new_image_shape, preserve_range=True)
    image_c2_downscaled = transform.resize(
        image_c2, new_image_shape, preserve_range=True)
    image_c3_downscaled = transform.resize(
        image_c3, new_image_shape, preserve_range=True)

    # Segmentation on channel 1:
    thresholded = segmentation_with_optimized_thresh(
        image_c1_downscaled, segm_pars['threshold'], segm_pars['max_iter'], segm_pars['fraction_range'])

    # Padding:
    image_c1_downscaled = image_padding(
        image_c1_downscaled, segm_pars['padding'])
    image_c2_downscaled = image_padding(
        image_c2_downscaled, segm_pars['padding'])
    image_c3_downscaled = image_padding(
        image_c3_downscaled, segm_pars['padding'])
    thresholded = image_padding(thresholded)

    # Clean up the segmentation with morphological transformations:
    thresholded = clean_up_segmented_image(thresholded, image_c1_downscaled, segm_pars['closing_r'],
                                           segm_pars['dilation_r'], segm_pars['mesh_radius'])

    # Apply the mask (thresholded image) to all channels:
    segmented_image_c1 = image_c1_downscaled*thresholded
    segmented_image_c2 = image_c2_downscaled*thresholded
    segmented_image_c3 = image_c3_downscaled*thresholded

    # Save the segmented images:
    image_file_names = [os.path.basename(filename_c1), os.path.basename(
        filename_c2), os.path.basename(filename_c3)]
    preprocessed_images = [segmented_image_c1,
                           segmented_image_c2, segmented_image_c3]
    new_file_names = aux_save_images(
        preprocessed_images, image_file_names, "Preprocessed_", destination_folder)

    return pd.Series([os.path.splitext(image_file_name)[0], new_file_names[0], new_file_names[1], new_file_names[2]])


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
            imsave(os.path.join(folder, filename), image)
            file_names_list.append(filename)
    else:
        filename = names
        imsave(os.path.join(folder, filename), image)
        file_names_list.append(filename)

    return list(file_names_list)


def segmentation_with_optimized_thresh(image, threshold=1.05, max_iter=200, fraction_range=[0.04, 0.05]):
    """
    Iteratively look for a global threshold that results in a segmented volume
    covering a given fraction of the volume of an ndimensional numpy array.

    Parameters
    ----------
    image : numpy array
        image to segment.
    threshold : float, optional
        starting value for thresholding, as a fraction of the average value of
        the array. The default is 1.05.
    max_iter : int, optional
        Dmaximum number of iterations. The default is 200.
    fraction_range : list, optional
        range of values of . The default is [0.025, 0.040].

    Returns
    -------
    thresholded_image: numpy array.
        a 3D binary mask of the segmented volume.

    """
    thresholded_image = image > threshold*np.mean(image)

    segm_fract = np.sum(thresholded_image)/thresholded_image.size
    step = 0.01

    n_iterations = 1

    while ((segm_fract > fraction_range[1]) or (segm_fract < fraction_range[0])) and n_iterations < max_iter:
        if segm_fract > fraction_range[1]:
            n_iterations += 1
            threshold += step
            thresholded_image = image > threshold*np.mean(image)
            segm_fract = np.sum(thresholded_image)/thresholded_image.size
            if segm_fract < fraction_range[0]:
                step = 0.1*step
        else:
            threshold -= step
            n_iterations += 1
            thresholded_image = image > threshold*np.mean(image)
            segm_fract = np.sum(thresholded_image)/thresholded_image.size
            if segm_fract > fraction_range[1]:
                step = 0.1*step

    return thresholded_image


def image_padding(image, padding=20, cval=0):
    """
    Padding of a 3D image with a constant value

    Parameters
    ----------
    image : 3D numpy array
        3D image.
    padding : int, optional
        extent of the padding. The default is 20.
    cval: int, optional
        constant value used for padding the image. The default is 0.

    Returns
    -------
    padded_image : 3d numpy array
        the padded image.

    """
    shape = np.asarray(np.shape(image))
    padded_image = np.ones(shape+2*padding)*cval
    padded_image[padding:-padding, padding:-padding, padding:-padding] = image
    return padded_image


def clean_up_segmented_image(binary_image, image, closing_r=4, dilation_r=8, mesh_radius=30):
    """
    This function refines the segmentation of a surface in a 3D image using
    morphological transformations (closing & dilation), selecting local maxima
    along the z direction, and fitting a mesh through the maxima to fill holes.

    Parameters
    ----------
    binary_image : 3d numpy array
        the preliminary segmentation of the image.
    image : 3d numpy array
        the original image.
    closing_r : int, optional
        closing radius. The default is 4.
    dilation_r : int, optional
        final dilation radius, which fixes the thickness of the segmented surface.
        The default is 5.

    Returns
    -------
    final_image : 3d numpy array
        a 3D binary mask of the segmented volume.

    """

    refined_binary_image = refine_surface_mask(binary_image, image, closing_r, surface_spacing_threshold = mesh_radius)

    surface_pcd, surface_pcd_values, _, _ = refine_surface_mask_with_mesh_interpolation(refined_binary_image, mesh_radius)

    final_image = pcd_to_image(surface_pcd, surface_pcd_values, binary_image.shape)

    final_image = morphology.dilation(final_image, morphology.ball(dilation_r))

    return final_image

def refine_surface_mask(binary_image, image, closing_r=4, surface_spacing_threshold=10):
    """
    Refines a binary surface mask by applying morphological operations and selecting local maxima
    along the z direction.

    Parameters
    ----------
    binary_image : 3D numpy array
        Preliminary segmentation of the image.
    image : 3D numpy array
        Original image.
    closing_r : int, optional
        Closing radius used to close small holes. Default is 4.
    surface_spacing_threshold : int, optional
        Threshold to reject local maxima that are too close along the axis. Default is 10.

    Returns
    -------
    refined_binary_image : 3D numpy array
        Refined binary mask of the segmented surface.

    """
    # Apply closing operation to close small holes:
    filled_binary_image = morphology.closing(binary_image, morphology.ball(closing_r))

    # Remove small objects from the mask:
    labelled_image = label(filled_binary_image)
    regions_properties = regionprops(labelled_image)
    max_object_size = max([region.area for region in regions_properties])
    biggest_objects_mask = morphology.remove_small_objects(
        labelled_image, min_size = max_object_size/100) > 0

    # Refine the mask selecting the local maxima of the image along the z direction:
    refined_binary_image = local_maxima_z(
        biggest_objects_mask * image, surface_spacing_threshold)

    return refined_binary_image

def refine_surface_mask_with_mesh_interpolation(binary_image, mesh_radius):
    """
    Refines a surface mask by interpolating a mesh to fill potential holes.

    Parameters
    ----------
    binary_image : ndarray
        Binary mask of the segmented surface.
    mesh_radius : float
        Radius used for mesh interpolation.

    Returns
    -------
    final_pcd : open3d.geometry.PointCloud
        Final point cloud representing the refined surface mask.
    final_pcd_values : ndarray
        Values associated with the final point cloud.
    original_pcd : open3d.geometry.PointCloud
        Original point cloud obtained from the binary mask.
    mesh : open3d.geometry.TriangleMesh
        Mesh interpolated from the downsampled point cloud.

    """

    # Create a point cloud object from the binary mask
    original_pcd, _ = image_to_pcd(binary_image)

    # Downsampling the point cloud to make the sampling more uniform
    downsampled_pcd = original_pcd.voxel_down_sample(voxel_size=4)

    # Fit a mesh through the cleaned points to fill potential holes
    downsampled_pcd.estimate_normals()
    downsampled_pcd.orient_normals_consistent_tangent_plane(k=30)
    downsampled_pcd.normals = o3d.utility.Vector3dVector(-np.asarray(downsampled_pcd.normals))
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
           downsampled_pcd, o3d.utility.DoubleVector([mesh_radius]))
    bbox = downsampled_pcd.get_axis_aligned_bounding_box()
    mesh_crop = mesh.crop(bbox)

    # Resample the mesh to obtain a point cloud
    mesh_area = mesh.get_surface_area()
    resampled_pcd = mesh_crop.sample_points_uniformly(number_of_points = int(2*mesh_area))

    # Merge the original point cloud and the resampled point cloud obtained from the mesh
    # to build the final point cloud representing the refined surface mask
    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points = o3d.utility.Vector3dVector(
        np.concatenate((original_pcd.points, resampled_pcd.points), axis=0))
    final_pcd_values = np.ones(np.asarray(final_pcd.points).shape[0])

    return final_pcd, final_pcd_values, original_pcd, mesh

def local_maxima_z(image, min_dist):
    """
    This function creates a mask of the image which selects the local maxima along
    the z direction which are separated by a minimum distance.

    Parameters
    ----------
    image : 3D numpy array
        3D image.
    min_dist : int
        minimum distance for local maxima.

    Returns
    -------
    mask_local_maxima_z : 3d numpy array
        a 3D binary mask of the segmented volume.
    """

    # maxima along z
    mask_local_maxima_z = np.zeros(image.shape)

    for i in range(image.shape[1]):
        for j in range(image.shape[2]):
            peaks, _ = find_peaks(image[:, i, j], distance=min_dist)
            for peak in peaks:
                mask_local_maxima_z[peak, i, j] = 1

    return mask_local_maxima_z

if __name__ == '__main__':

    ## %matplotlib qt  ##
    read_data_folder = "../../data/01_raw"
    destination_folder = "../../data/02_preprocessed"
    preprocess_and_segment_images(
        read_data_folder, destination_folder, downscaling=[1, 2.5, 2.5], bit_depth=12)
