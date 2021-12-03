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


def registration_of_abdomens_3D(
        preprocessed_data_df, preprocessed_folder, reference_fly_filename,\
        abdomen_mask_filename, destination_folder):
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

    for f in os.listdir(destination_folder):
        os.remove(os.path.join(destination_folder, f))

    print("Registration of 3D stacks in progress:")
    reference_fly = io.imread(reference_fly_filename)
    abdomen_mask = io.imread(abdomen_mask_filename)

    preprocessed_data_df[["filename_fl", "filename_tl"]] = preprocessed_data_df.progress_apply(lambda row: \
    register_and_save(row["filename_fl"], row["filename_tl"], preprocessed_folder, reference_fly, abdomen_mask, destination_folder), axis=1)
    
    preprocessed_data_df["folder"] = destination_folder
    preprocessed_data_df.to_excel(os.path.join(destination_folder,'DatasetInformation.xlsx'))

    return


def register_and_save(filename_fl, filename_tl, folder, reference_fly, abdomen_mask, destination_folder):
    
    filename_fl = os.path.join(folder,filename_fl)
    filename_tl = os.path.join(folder,filename_tl)
    
    try:
        Source_Image =  io.imread(filename_fl)
        Source_TL  = io.imread(filename_tl)
    except:
        print("File not found!")
        Source_Image = float("NaN")
        Source_TL  = float("NaN")

    source, Source_values = image_to_pcd(Source_Image)
    source_TL, Source_values_TL = image_to_pcd(Source_TL)

    target, Target_values = image_to_pcd(reference_fly)
    mask_pcd, mask_values = image_to_pcd(abdomen_mask)

    source = prealignment(source,target)

    # Refine registration with ICP:
    
    result_icp = refine_registration(source, target, 50)
    after_icp = copy.deepcopy(source)
    after_icp.transform(result_icp.transformation)
    
    registered_source_image = pcd_to_image(after_icp,Source_values,reference_fly.shape)
    registered_source_image_TL = pcd_to_image(after_icp,Source_values_TL,reference_fly.shape)
    
    #draw_registration_result(after_icp, target, np.eye(4))
    
    # Apply mask to select only the fly abdomen:
    registered_source_image = pcd_to_image(after_icp,Source_values,reference_fly.shape)
    final_image = registered_source_image*abdomen_mask
    final_image_TL = registered_source_image_TL*abdomen_mask
    
    #final_pcd, final_values = image_to_pcd(final_image)
    #final_pcd_TL, final_values_TL = image_to_pcd(final_image_TL)

    image_file_names = [os.path.basename(filename_fl), os.path.basename(filename_tl)]
    registered_images = [final_image, final_image_TL]
    new_file_names = aux_save_images(registered_images, image_file_names, destination_folder)
    
    return pd.Series([new_file_names[0], new_file_names[1]])

    
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

def prealignment(pcd, target_pcd):
    target_mean, target_cov = target_pcd.compute_mean_and_covariance()
    pcd_mean, pcd_cov = pcd.compute_mean_and_covariance()
    _ , pcd_eigv = np.linalg.eig(pcd_cov)

    # transform to a base given by the first eigenvector (a,b,c), (0,0,1) and their vector product: (b,-a,0)
    b = pcd_eigv[1,0]
    c = pcd_eigv[2,0]
    b, c = b/(b**2+c**2)**0.5, c/(b**2+c**2)**0.5
    temp = np.asarray([[1.0, 0.0, 0.0], [0.0, b, c], [0.0, c, -b]])
    transformation_pcd_1 = np.eye(4)
    transformation_pcd_1[:3, :3] = np.linalg.inv(temp)
    
    temp = np.asarray([[1.0, 0.0, 0.0], [0.0, -b, -c], [0.0, -c, b]])
    transformation_pcd_2 = np.eye(4)
    transformation_pcd_2[:3, :3] = np.linalg.inv(temp)
    
    test_1 = copy.deepcopy(pcd)
    test_1.translate(-pcd_mean)
    test_1.transform(transformation_pcd_1)
    test_1.translate(target_mean)
    
    distances = test_1.compute_point_cloud_distance(target_pcd)
    dist_1 = np.sum(np.asarray(distances))
    
    test_2 = copy.deepcopy(pcd)
    test_2.translate(-pcd_mean)
    test_2.transform(transformation_pcd_2)
    test_2.translate(target_mean)
    
    distances = test_2.compute_point_cloud_distance(target_pcd)
    dist_2 = np.sum(np.asarray(distances))
    
    if dist_1 < dist_2:
        return test_1
    else:
        return test_2
    
def refine_registration(source, target, threshold):
    result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False))
    return result

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
    
    #read_folder = "/media/ceolin/Data/Lab Gompel/Projects/Fly_Abdomens/data/02_preprocessed"
    #destination_folder = "/media/ceolin/Data/Lab Gompel/Projects/Fly_Abdomens/data/03_registered"
    reference_fly_filename = "/media/ceolin/Data/Lab Gompel/Projects/Fly_Abdomens/data/References_and_masks/Aligned_Reference.tif"
    abdomen_mask_file = "/media/ceolin/Data/Lab Gompel/Projects/Fly_Abdomens/data/References_and_masks/Abdomen_Mask.tif"
    image_file = "/media/ceolin/Data/Lab Gompel/Projects/Fly_Abdomens/data/02_preprocessed/Preprocessed_C1-20190126 - Experiment_A0B0_female1.tif"
    image_file_2 = "/media/ceolin/Data/Lab Gompel/Projects/Fly_Abdomens/data/02_preprocessed/Preprocessed_C2-20190126 - Experiment_A0B0_female1.tif"

    #df_name = "DatasetInformation.xlsx"
    #df = pd.read_excel(os.path.join(read_folder,df_name))
    #registration_of_abdomens_3D(df, read_folder, reference_fly_filename, abdomen_mask_file, destination_folder)
    Reference_Image = io.imread(reference_fly_filename)
    Abdomen_Mask = io.imread(abdomen_mask_file)

    Source_Image = io.imread(image_file)
    
    Source_TL = io.imread(image_file_2)
    source, Source_values = image_to_pcd(Source_Image)
    source_TL, Source_values_TL = image_to_pcd(Source_TL)

    target, Target_values = image_to_pcd(Reference_Image)

    mask_pcd, mask_values = image_to_pcd(Abdomen_Mask)
    source = prealignment(source,target)
    # Refine registration with ICP:
    
    result_icp = refine_registration(source, target, 50)
    after_icp = copy.deepcopy(source)
    after_icp.transform(result_icp.transformation)
    
    registered_source_image = pcd_to_image(after_icp,Source_values,Reference_Image.shape)
    registered_source_image_TL = pcd_to_image(after_icp,Source_values_TL,Reference_Image.shape)
    
    draw_registration_result(after_icp, target, np.eye(4))
    
    # Apply mask to select only the fly abdomen:
    registered_source_image = pcd_to_image(after_icp,Source_values,Reference_Image.shape)
    final_image = registered_source_image*Abdomen_Mask
    final_image_TL = registered_source_image_TL*Abdomen_Mask
    
    final_pcd, final_values = image_to_pcd(final_image)
    final_pcd_TL, final_values_TL = image_to_pcd(final_image_TL)
