#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function(s) to manually register a series of 3D image stacks with multiple separate
channels and save the registered images.

@author: ceolin
"""

import sys
import os
import copy
from io import StringIO
import pandas as pd
from skimage import io
import numpy as np
import open3d as o3d
from tifffile import imsave
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

if __name__ == '__main__':
    from aux_pcd_functions import pcd_to_image, image_to_pcd
else:
    from src.aux_pcd_functions import pcd_to_image, image_to_pcd


def run_manual_registration(
        preprocessed_data_df, preprocessed_folder, reference_image_filename,
        destination_folder, only_on_new_files=False,
        database_filename='DatasetInformation.xlsx'):
    """
    This function loops through all the preprocessed images, starts the manual
    registration gui, saves the output registered images in the destination
    folder and updates or creates the excel file with the information about each
    image.

    Parameters
    ----------
    preprocessed_data_df : pandas dataframe
        dataframe listing all preprocessed image files and the corresponding
        genotype and additional info.
    preprocessed_folder : str
        folder were preprocessed images are saved.
    reference_image_filename : str
        path to the reference image to use in the registration.
    destination_folder : str
        folder were registered images will be saved.
    only_on_new_files : bool, optional
        flag to skip images that ave been already registered. The default is False.
    database_filename : str, optional
        name of the excel file(s) included with the raw data.
        The default is 'DatasetInformation.xlsx'.

    Returns
    -------
    None.

    """

    tqdm.pandas()
    preprocessed_folder = os.path.join(preprocessed_folder, '')

    # clean the destination directory:
    if not only_on_new_files:
        for file in os.listdir(destination_folder):
            os.remove(os.path.join(destination_folder, file))

    try:
        registered_df = pd.read_excel(os.path.join(
            destination_folder, database_filename))
    except FileNotFoundError:
        registered_df = pd.DataFrame(columns=[
                                     "image file name", "type", "folder", "experiment", "filename_c1", "filename_c2", "filename_c3"])

    print("Registration of 3D stacks in progress:")
    reference_image = io.imread(reference_image_filename)

    with tqdm(total=preprocessed_data_df.shape[0]) as pbar:
        # loop explicitly on all files to be able to save the excel file after
        # each step.
        for _, row in preprocessed_data_df.iterrows():

            if os.path.splitext(row["image file name"])[0] in registered_df['experiment'].values:
                pass

            else:
                registered_filenames = register_and_save_image_stack(
                    row["image file name"], preprocessed_folder, reference_image, destination_folder)

                registered_df = pd.concat(
                    [registered_df, row.to_frame().transpose()], join="outer", ignore_index=True)

                #temp_df = temp_df[registered_df.columns]

                #registered_df = temp_df

                registered_df.loc[registered_df['image file name']
                                  == row["image file name"], "filename_c1"] = registered_filenames[0]
                registered_df.loc[registered_df['image file name']
                                  == row["image file name"], "filename_c2"] = registered_filenames[1]
                registered_df.loc[registered_df['image file name']
                                  == row["image file name"], "filename_c3"] = registered_filenames[2]
                registered_df.loc[registered_df['image file name']
                                  == row["image file name"], "folder"] = destination_folder
                registered_df = registered_df.reset_index(drop=True)

                registered_df.to_excel(os.path.join(
                    destination_folder, 'DatasetInformation.xlsx'), index=False)

            pbar.update(1)

        pbar.close()

    return


def register_and_save_image_stack(image_file_name, input_folder, reference_image, destination_folder):
    """
    This function is used to manually register the three channels of one
    3D image stack and save the registered images in the destination folder.

    Parameters
    ----------
    image_file_name : str
        original file name of the multichannel image stack.
    input_folder : str
    reference_image : 3d numpy array.
        prealigned reference image.
    destination_folder : str
        destination folder for registered images.

    Returns
    -------
    None.

    """

    filename_c1 = os.path.join(
        input_folder, 'Preprocessed_C1-'+image_file_name)
    filename_c2 = os.path.join(
        input_folder, 'Preprocessed_C2-'+image_file_name)
    filename_c3 = os.path.join(
        input_folder, 'Preprocessed_C3-'+image_file_name)

    try:
        image_src_c1 = io.imread(filename_c1).astype(np.uint16)
        image_src_c2 = io.imread(filename_c2).astype(np.uint16)
        image_src_c3 = io.imread(filename_c3).astype(np.uint16)

    except FileNotFoundError:
        print("File not found: " + image_file_name)
        return None

    # Convert images in point clouds:
    pcd_src_c1, src_values_c1 = image_to_pcd(image_src_c1)
    pcd_target, target_values = image_to_pcd(reference_image)

    # Open the interface for manual registration:
    transformation = manual_registration(
        pcd_src_c1, src_values_c1, pcd_target, target_values)
    
    # Draw the result:
    pcd_src_c1.transform(transformation)
    draw_registration_result(pcd_src_c1, pcd_target)
  
    # Apply the transformation:
    final_shape = reference_image.shape
    arguments = [(image_src_c1, transformation, final_shape),
                 (image_src_c1, transformation, final_shape),
                 (image_src_c1, transformation, final_shape)] 
    
    with ProcessPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(aux_apply_registration_for_pool, arguments))
    
    registered_source_image_c1  = results[0]
    registered_source_image_c2  = results[1]
    registered_source_image_c3  = results[2]

    # Prepare filenames and save the registered images:
    filename_c1 = 'C1-'+image_file_name
    filename_c2 = 'C2-'+image_file_name
    filename_c3 = 'C3-'+image_file_name

    registered_source_image_c1 = registered_source_image_c1.astype(np.uint16)
    registered_source_image_c2 = registered_source_image_c2.astype(np.uint16)
    registered_source_image_c3 = registered_source_image_c3.astype(np.uint16)

    image_file_names = [filename_c1, filename_c2, filename_c3]
    registered_images = [registered_source_image_c1,
                         registered_source_image_c2, registered_source_image_c3]
    new_file_names = aux_save_images(
        registered_images, image_file_names, "Registered_", destination_folder)

    return [new_file_names[0], new_file_names[1], new_file_names[2]]

def aux_apply_registration_for_pool(arguments):
    registered_source_image = aux_apply_registration(arguments[0], arguments[1], arguments[2])
    return registered_source_image


def aux_apply_registration(image_src, transformation, shape):
    """
    Apply registration transformation to an input image and return the registered image.

    Parameters:
        image_src (numpy.ndarray): The source image as a 2D NumPy array.
        transformation (open3d.geometry.Geometry3D): An Open3D Geometry3D object representing the
                               transformation to be applied to the source image.
        shape (tuple): A tuple (height, width) representing the desired shape of the registered image.

    Returns:
        numpy.ndarray: The registered source image as a 2D NumPy array.

    Raises:
        ValueError: If the input `image_src` is not a valid 2D NumPy array.
        ValueError: If the input `shape` is not a valid tuple of two positive integers (height, width).
        ValueError: If the input `transformation` is not a valid Open3D Geometry3D object.

    Notes:
        - The resulting image will be of the size specified in the 'shape' parameter.
    """
    # Create point clouds with higher sampling for rotation:
    pcd_src, src_values = image_to_pcd(image_src, upscale = 2)
    # Apply the transformation on all channels:
    pcd_src.transform(transformation)
    # Convert the registered point clouds to image stacks:
    registered_source_image = pcd_to_image(pcd_src, src_values, shape)
    return registered_source_image

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


def draw_registration_result(blue_pcd, yellow_pcd):
    """
    This function draws two point clouds in blue and yellow

    Parameters
    ----------
    source : point cloud
    target : point cloud

    Returns
    -------
    None.

    """
    blue_pcd_temp = copy.deepcopy(blue_pcd)
    yellow_pcd_temp = copy.deepcopy(yellow_pcd)
    blue_pcd_temp.paint_uniform_color([1, 0.706, 0])
    yellow_pcd_temp.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([blue_pcd_temp, yellow_pcd_temp])

    return


def manual_registration(source, source_values, target, target_values):
    """
    This function shows two coloured pcd objects in a gui, allows the user to
    pick corresponding points on each of them and calculate the affine
    transformation mapping the source points cloud on the target points cloud.

    Parameters
    ----------
    source : pcd
    source_values : numpy array
    target : pcd
    target_values : numpy array

    Returns
    -------
    transformation: open3d transformation
    the affine transformation.

    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)

    bw_colors = 10*(source_values-np.min(source_values))/(np.max(source_values)-np.min(source_values))
    source_temp.colors = o3d.utility.Vector3dVector(
        np.asarray([bw_colors, bw_colors, bw_colors]).T)

    bw_colors = 10*(target_values-np.min(target_values))/(np.max(target_values)-np.min(target_values))
    target_temp.colors = o3d.utility.Vector3dVector(
        np.asarray([bw_colors, bw_colors, bw_colors]).T)

    # pick points from two point clouds and builds correspondences
    picked_id_source = pick_points(source_temp)
    picked_id_target = pick_points(target_temp)

    assert (len(picked_id_source) >= 3 and len(picked_id_target) >= 3),\
        " ERROR: not enough points."
    assert (len(picked_id_source) == len(picked_id_target)),\
        " ERROR: different number of points in the reference and target images."
    corr = np.zeros((len(picked_id_source), 2))
    corr[:, 0] = picked_id_source
    corr[:, 1] = picked_id_target

    # estimate transformation:
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint(
        with_scaling=True)
    transformation = p2p.compute_transformation(
        source, target, o3d.utility.Vector2iVector(corr))

    return transformation


def pick_points(pcd, point_size = None, window_name = "pick points"):
    """
    This function visualizes a point cloud object in  a gui and allows the user
    to select a series of points on the object.
    Returns the coordinates of the selected points.

    Parameters
    ----------
    pcd : point cloud object
    point_size : int, size of the representation of a point
    window_name : str, name of the window

    Returns
    -------
    selected points: point cloud object

    """

    # These are used to suppress the printed output from Open3D while picking points:
    stdout_old = sys.stdout
    sys.stdout = StringIO()
    # Create Visualizer with editing:
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name = window_name)
    vis.add_geometry(pcd)
    if point_size:
        renderer_opts = vis.get_render_option()
        renderer_opts.point_size = np.array(point_size)
    # user picks points
    result = vis.run()
    vis.destroy_window()
    # This restores the output:
    sys.stdout = stdout_old
    return vis.get_picked_points()


def refine_registration_point_to_plane(source, target, threshold, downsampling_radius):
    """
    Parameters
    ----------
    source : pcd object
        source point cloud.
    target : pcd object
        target point cloud.
    threshold : int
        maximum distance between corresponding points.
    downsampling_radius : int
        radius used to downsample the point clouds.

    Returns
    -------
    result : o3d transformation object
        estimated transformation.

    """
    # downsampling to accelerate computation:
    source_down = source.voxel_down_sample(downsampling_radius)
    target_down = target.voxel_down_sample(downsampling_radius)

    # computing the normals for each point:
    source_down.estimate_normals()
    source_down.orient_normals_consistent_tangent_plane(k=30)

    target_down.estimate_normals()
    target_down.orient_normals_consistent_tangent_plane(k=30)

    result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500))

    return result


if __name__ == '__main__':

    read_folder = "../test_dataset/02_preprocessed"
    destination_folder = "../test_dataset/03_registered"

    reference_fly_filename = "../test_dataset/References_and_masks/C1_Reference_iso.tiff"
    

    df_name = "DatasetInformation.xlsx"

    preprocessed_df = pd.read_excel(os.path.join(read_folder, df_name))
    run_manual_registration(preprocessed_df, read_folder, reference_fly_filename,
                            destination_folder, only_on_new_files=True)
