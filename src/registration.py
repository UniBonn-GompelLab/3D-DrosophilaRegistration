#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function(s) to register 3D stacks of fly abdomens

@author: ceolin
"""

import pandas as pd
import os
from io import StringIO
from skimage import io
import numpy as np
import open3d as o3d
import copy
from tifffile import imsave
from tqdm import tqdm
import sys

if __name__ == '__main__':
    from aux_pcd_functions  import pcd_to_image, image_to_pcd
else:
    from src.aux_pcd_functions  import pcd_to_image, image_to_pcd
    
    
def registration_of_abdomens_3D(
        preprocessed_data_df, preprocessed_folder, reference_fly_filename,\
        abdomen_mask_filename, destination_folder, only_on_new_files = False, \
        database_filename = 'DatasetInformation.xlsx'):
    '''
    Parameters
    ----------
    preprocessed_data_df : pandas dataframe 
        dataframe listing all preprocessed image files and the corresponding
        genotype.
    
    preprocessed_folder: str 
        folder were preprocessed images are saved.
    
    reference_fly_filename: str
        path to the reference abdomen file.
    
    abdomen_mask_filename: str
        path to the mask to sleect only the abdomen.
    
    destination_folder: str
        folder were registered images will be saved.

    '''

    tqdm.pandas()
    preprocessed_folder = os.path.join(preprocessed_folder,'')
    

    # clean the destination directory:
    if only_on_new_files == False:
        for f in os.listdir(destination_folder):
            os.remove(os.path.join(destination_folder, f))
            
    try: 
        DatasetInfoRegistered = pd.read_excel(os.path.join(destination_folder,database_filename))
    except:
        DatasetInfoRegistered = pd.DataFrame(columns = ["image file name", "quality", "experiment", "construct",  "folder", "filename_gfp", "filename_dsred", "filename_tl"])


    print("Registration of 3D stacks in progress:")
    reference_fly = io.imread(reference_fly_filename)
    
    with tqdm(total=preprocessed_data_df.shape[0]) as pbar:    
        
        for index, row in preprocessed_data_df.iterrows():
        
            result = register_and_save(row["image file name"], row["filename_gfp"], row["filename_dsred"], row["filename_tl"], preprocessed_folder, reference_fly, destination_folder, DatasetInfoRegistered)
            
            if result is not None:
                temp_df = pd.concat([DatasetInfoRegistered, row.to_frame().transpose()], join = "outer", ignore_index=True)
                temp_df = temp_df[DatasetInfoRegistered.columns]
                DatasetInfoRegistered = temp_df
                DatasetInfoRegistered.loc[DatasetInfoRegistered['image file name'] == row["image file name"], "filename_gfp"]= result[0]
                DatasetInfoRegistered.loc[DatasetInfoRegistered['image file name'] == row["image file name"], "filename_dsred"] = result[1]
                DatasetInfoRegistered.loc[DatasetInfoRegistered['image file name'] == row["image file name"], "filename_tl"]  = result[2]
                DatasetInfoRegistered.loc[DatasetInfoRegistered['image file name'] == row["image file name"], "folder"] = destination_folder
                DatasetInfoRegistered = DatasetInfoRegistered.reset_index(drop = True)
                DatasetInfoRegistered.to_excel(os.path.join(destination_folder,'DatasetInformation.xlsx'), index = False)
            
            pbar.update(1)
            
        pbar.close()

    return


def register_and_save(image_file_name, filename_gfp, filename_dsred, filename_tl, folder, reference_fly, destination_folder, DatasetInfoRegistered):
    """
    This function is used to manually register the three channels of one 
    3d image stack, save the registered images in the destination folder 
    and update the dataframe with the image information.

    Parameters
    ----------
    image_file_name : str
        original file name of the multichannel image stack.
    filename_gfp : str
        file name of the gfp channel.
    filename_dsred : str
        file name of the dsred channel.
    filename_tl : str
        file name of the transmitted light channel.
    folder : str
        folder where images are located.
    reference_fly : 3d numpy array.
        prealigned reference image, gfp channel only.
    destination_folder : str
        destination folder for registered images.
    DatasetInfoRegistered : pandas dataframe
        dataframe listing registered image files and the corresponding
        genotype.

    Returns
    -------
    None.

    """ 
    
    if os.path.splitext(image_file_name)[0] in DatasetInfoRegistered['experiment'].values:
        return None

    try:
        Source_gfp   = io.imread( os.path.join(folder,filename_gfp) )
        Source_dsred = io.imread( os.path.join(folder,filename_dsred) )
        Source_tl    = io.imread( os.path.join(folder,filename_tl) )
    except:
        print("File not found!")
        return None

    source_gfp, source_values_gfp = image_to_pcd(Source_gfp)
    source_dsred, source_values_dsred = image_to_pcd(Source_dsred)
    source_tl, source_values_tl = image_to_pcd(Source_tl)
    target, target_values = image_to_pcd(reference_fly)

    # Open the interface for manual registration:
    transformation = _manual_registration(source_gfp, source_values_gfp, target, target_values)
    source_gfp.transform(transformation)
    source_dsred.transform(transformation)
    source_tl.transform(transformation)
    
    # Draw the results:
    draw_registration_result(source_gfp, target)
    
    # Convert the registered point clouds to image stacks:
    registered_source_image = pcd_to_image(source_gfp, source_values_gfp, reference_fly.shape)
    registered_source_image_dsred = pcd_to_image(source_dsred, source_values_dsred, reference_fly.shape)
    registered_source_image_tl = pcd_to_image(source_tl, source_values_tl, reference_fly.shape)

    # Prepare filenames and save the registered images:
    filename_GFP = os.path.join(folder,'C1-'+image_file_name)
    filename_DsRed = os.path.join(folder,'C2-'+image_file_name)
    filename_TL = os.path.join(folder,'C3-'+image_file_name)

    image_file_names = [os.path.basename(filename_GFP), os.path.basename(filename_DsRed), os.path.basename(filename_TL)]
    registered_images = [registered_source_image, registered_source_image_dsred, registered_source_image_tl]
    new_file_names = aux_save_images(registered_images, image_file_names, "Registered_", destination_folder)
    
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

def draw_registration_result(source, target):
    """
    Draw two point clouds in blue and yellow
    
    Parameters
    ----------
    source : point cloud
    target : point cloud

    Returns
    -------
    None.

    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([source_temp, target_temp])
    
    return

def _manual_registration(source, source_values, target, target_values):
    """
    Parameters
    ----------
    source : pcd
    source_values : numpy array
    target : pcd
    target_values : numpy array
    
    Returns
    -------
    transformation: open3d transformation
    the affine transformation mapping the source points cloud on the target points cloud.

    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    
    bw_colors = 10*(source_values-np.min(source_values))/np.max(source_values)
    source_temp.colors = o3d.utility.Vector3dVector(np.asarray([bw_colors, bw_colors, bw_colors]).T)

    bw_colors = 10*(target_values-np.min(target_values))/np.max(target_values)
    target_temp.colors = o3d.utility.Vector3dVector(np.asarray([bw_colors, bw_colors, bw_colors]).T)

    # pick points from two point clouds and builds correspondences
    picked_id_source = pick_points(source_temp)
    picked_id_target = pick_points(target_temp)
    
    assert (len(picked_id_source) >= 3 and len(picked_id_target) >= 3)
    assert (len(picked_id_source) == len(picked_id_target))
    corr = np.zeros((len(picked_id_source), 2))
    corr[:, 0] = picked_id_source
    corr[:, 1] = picked_id_target

    # estimate transformation:
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)
    transformation = p2p.compute_transformation(source, target, o3d.utility.Vector2iVector(corr))

    return transformation


def pick_points(pcd):
    """
    Visualize a point cloud object and allows the user to select a series of points
    on the object. Returns the coordinates of the selected points.

    Parameters
    ----------
    pcd : point cloud object

    Returns
    -------
    selected points (pcd).

    """
    
    # These are used to suppress the printed output from Open3D while picking points:
    stdout_old = sys.stdout
    sys.stdout = StringIO()
    # Create Visualizer with editing:
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    # user picks points
    result = vis.run()  
    vis.destroy_window()
    # This restores the output:
    sys.stdout = stdout_old
    return vis.get_picked_points()

def refine_registration_PointToPlane(source, target, threshold, downsampling_radius):
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

    
    read_folder = "../../data/02_preprocessed"
    destination_folder = "../../data/03_registered"
    
    reference_fly_filename = "../../data/References_and_masks/C1_Reference_iso.tif"
    abdomen_mask_file = "../../data/References_and_masks/Reference_abdomen_mask_iso.tif"

    df_name = "DatasetInformation.xlsx"
    
    preprocessed_df = pd.read_excel(os.path.join(read_folder,df_name))
    registration_of_abdomens_3D(preprocessed_df, read_folder, reference_fly_filename, abdomen_mask_file, destination_folder, only_on_new_files = True)
