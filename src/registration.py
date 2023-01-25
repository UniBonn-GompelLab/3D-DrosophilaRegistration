#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function(s) to register 3d stacks of fly abdomens

@author: ceolin
@last_update: September 2022
"""

import pandas as pd
import os
from io import StringIO
from skimage import io, transform
import numpy as np
import open3d as o3d
import copy
#import napari
from skimage import morphology
from skimage.measure import label, regionprops, block_reduce
from scipy import stats, ndimage
import matplotlib.pyplot as plt
from tifffile import imsave
from tqdm import tqdm
import sys
import os

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
        dataframe listing all preprocessed image files in the column 'image file name'
        and to which construct the images correspond.
    
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
    abdomen_mask = io.imread(abdomen_mask_filename)
    
    
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
    
    filename_gfp = os.path.join(folder,filename_gfp)
    filename_dsred = os.path.join(folder,filename_dsred)
    filename_tl = os.path.join(folder,filename_tl)
    
    if os.path.splitext(image_file_name)[0] in DatasetInfoRegistered['experiment'].values:
        return None

    try:
        Source_gfp =  io.imread(filename_gfp)
        Source_dsred  = io.imread(filename_dsred)
        Source_tl  = io.imread(filename_tl)
    except:
        print("File not found!")
        return None

    #print("******************")
    #print(image_file_name)
    source, source_values = image_to_pcd(Source_gfp)
    source_dsred, source_values_dsred = image_to_pcd(Source_dsred)
    source_tl, source_values_tl = image_to_pcd(Source_tl)

    target, target_values = image_to_pcd(reference_fly)

    #source = prealignment(source,target)
    source = _manual_registration(source, source_values, target, target_values)
    
    # Refine registration with point to plane ICP:
    #result_ptp_icp = refine_registration_PointToPlane(source, target, threshold = 50 , downsampling_radius = 5)
    after_icp = copy.deepcopy(source)
    #after_icp.transform(result_ptp_icp.transformation)
    #draw_registration_result(after_icp, target, np.identity(4))
    
    registered_source_image = pcd_to_image(after_icp, source_values, reference_fly.shape)
    registered_source_image_dsred = pcd_to_image(after_icp, source_values_dsred, reference_fly.shape)
    registered_source_image_tl = pcd_to_image(after_icp, source_values_tl, reference_fly.shape)

    image_file_names = [os.path.basename(filename_gfp), os.path.basename(filename_dsred), os.path.basename(filename_tl)]
    registered_images = [registered_source_image, registered_source_image_dsred, registered_source_image_tl]
    new_file_names = aux_save_images(registered_images, image_file_names, destination_folder)
    
    return pd.Series([new_file_names[0], new_file_names[1], new_file_names[2]])

    
def aux_save_images(images,names,folder):
    file_names_list = list()
    if isinstance(images, list):
        for count, image in enumerate(images):
            filename = names[count].replace("Preprocessed_","Registered_" )
            imsave(os.path.join(folder,filename), image)
            file_names_list.append(filename)
    else:
        filename = names
        imsave(os.path.join(folder,filename), image)
        file_names_list.append(filename)
    return list(file_names_list)

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def _manual_registration(source, source_values, target, target_values):
    
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    
    bw_colors = 3*source_values/np.max(source_values)
    source_temp.colors = o3d.utility.Vector3dVector(np.asarray([bw_colors, bw_colors, bw_colors]).T)

    bw_colors = 3*target_values/np.max(target_values)
    target_temp.colors = o3d.utility.Vector3dVector(np.asarray([bw_colors, bw_colors, bw_colors]).T)

    #target_temp = target_temp.uniform_down_sample(every_k_points=10)
    #source_temp = source_temp.uniform_down_sample(every_k_points=10)
    
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
    trans_init = p2p.compute_transformation(source, target, o3d.utility.Vector2iVector(corr))
    source_transformed = copy.deepcopy(source)
    source_transformed.transform(trans_init)
    
    draw_registration_result(source_transformed, target, np.identity(4))
    
    return source_transformed


def pick_points(pcd):
    # These are used to suppress the printed output from Open3D whil epicking points:
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

# def refine_registration(source, target, threshold):
#     result = o3d.pipelines.registration.registration_icp(
#         source, target, threshold, np.identity(4),
#         o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False))
#     return result

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



# def prealignment(source, target_pcd, force_flipping = False):
#     target_mean, target_cov = target_pcd.compute_mean_and_covariance()
#     source_mean, source_cov = source.compute_mean_and_covariance()
#     _ , source_eigv = np.linalg.eig(source_cov)

#     # transform to a base given by the x-y projection of the first eigenvector (0,b,c), 
#     # the z-direction (0,0,1) and their vector product: (0,c,-b)
    
#     b = source_eigv[1,0]
#     c = source_eigv[2,0]
#     b, c = b/(b**2+c**2)**0.5, c/(b**2+c**2)**0.5
#     temp = np.asarray([[1.0, 0.0, 0.0], [0.0, b, c], [0.0, c, -b]])
#     transformation_1 = np.eye(4)
#     transformation_1[:3, :3] = np.linalg.inv(temp)
    
#     temp = np.asarray([[1.0, 0.0, 0.0], [0.0, -b, -c], [0.0, -c, b]])
#     transformation_2 = np.eye(4)
#     transformation_2[:3, :3] = np.linalg.inv(temp)
    
#     test_1 = copy.deepcopy(source)
#     test_1.translate(-source_mean)
#     test_1.transform(transformation_1)
#     test_1.translate(target_mean)
    
#     distances = test_1.compute_point_cloud_distance(target_pcd)
#     dist_1 = np.sum(np.asarray(distances))
    
#     test_2 = copy.deepcopy(source)
#     test_2.translate(-source_mean)
#     test_2.transform(transformation_2)
#     test_2.translate(target_mean)
    
#     distances = test_2.compute_point_cloud_distance(target_pcd)
#     dist_2 = np.sum(np.asarray(distances))

    
#     orientation = _choose_best_orientation_gui(test_1, test_2, target_pcd)

#     if orientation == 1:
#         return test_1
#     else:
#         return test_2
    
# def _choose_best_orientation_gui(source_1, source_2, target):

#     source_1_temp = copy.deepcopy(source_1)
#     source_2_temp = copy.deepcopy(source_2)
#     target_temp = copy.deepcopy(target)
    
#     source_1_temp =  source_1_temp.voxel_down_sample(voxel_size=10)
#     source_2_temp =  source_2_temp.voxel_down_sample(voxel_size=10)
#     target_temp =  target_temp.voxel_down_sample(voxel_size=10)
    
#     source_1_temp.paint_uniform_color([1, 0.706, 0])
#     source_2_temp.paint_uniform_color([1, 0.606, 0])
#     target_temp.paint_uniform_color([0, 0.651, 0.929])
    
    
#     vis = o3d.visualization.VisualizerWithKeyCallback()
    
#     def press_1(vis, action, mods):
#         nonlocal orientation
#         orientation = 1
#         try:
#             vis.remove_geometry(source_2_temp)
#         except:
#             pass
            
#         vis.add_geometry(source_1_temp)
#         #vis.update_geometry()
#         return
    
#     def press_2(vis, action, mods):
#         nonlocal orientation
#         orientation = 2
#         try:
#             vis.remove_geometry(source_1_temp)
#         except:
#             pass
#         vis.add_geometry(source_2_temp)
#         #vis.update_geometry()
#         return

#     # key_action_callback will be triggered when there's a keyboard press, release or repeat event
#     # for on of the following keys, correspondence based on GLFW keyboard keys 
#     # (https://www.glfw.org/docs/latest/group__keys.html) :
#     vis.register_key_action_callback(49, press_1)  # pressing key 1
#     vis.register_key_action_callback(50, press_2)  # pressing key 2

#     vis.create_window()
    
#     orientation = 1
#     vis.add_geometry(source_1_temp)
#     vis.add_geometry(target_temp)
#     vis.run()
    
#     return orientation





if __name__ == '__main__':

    
    read_folder = "/media/ceolin/Data/Lab Gompel/Projects/Fly_Abdomens/data_2/02_preprocessed"
    destination_folder = "/media/ceolin/Data/Lab Gompel/Projects/Fly_Abdomens/data_2/03_registered"
    
    reference_fly_filename = "/media/ceolin/Data/Lab Gompel/Projects/Fly_Abdomens/data_2/References_and_masks/Reference_abdomen_2_2_2.tif"
    abdomen_mask_file = "/media/ceolin/Data/Lab Gompel/Projects/Fly_Abdomens/data_2/References_and_masks/Reference_abdomen_mask_thick_2_2_2.tif"
    
    df_name = "DatasetInformation.xlsx"
    
    preprocessed_df = pd.read_excel(os.path.join(read_folder,df_name))
    registration_of_abdomens_3D(preprocessed_df, read_folder, reference_fly_filename, abdomen_mask_file, destination_folder, only_on_new_files = True)
