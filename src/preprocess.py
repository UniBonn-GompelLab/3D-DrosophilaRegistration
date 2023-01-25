##################################################
## Preprocessing and segmentation of fly abdomens
##################################################
## Author: Stefano
## Version: November 2021
##################################################

import pandas as pd
import os
from skimage import io, transform
import numpy as np
import open3d as o3d
import copy
from skimage import morphology
from skimage.measure import label, regionprops, block_reduce
from scipy import stats
import matplotlib.pyplot as plt
from tifffile import imsave
from tqdm import tqdm
if __name__ == '__main__':
    from aux_pcd_functions  import pcd_to_image, image_to_pcd
else:
    from src.aux_pcd_functions  import pcd_to_image, image_to_pcd

def preprocess_and_segment_images(
    
    read_folder, destination_folder, binning=(18,6,6), bit_depth=8, only_on_new_files = True,\
    database_filename = 'DatasetInformation.xlsx'):
    
    '''
    Parameters
    ----------
    read_folder : str
        path of the folder containing the raw data.
    destination_folder : str
        path of the folder where preprocessed images will be saved.
    binning : (int, int, int)
        binning to apply to the images along z, x, y, in pixels.
    bit_depth: int
        bit depth of raw data
    database_filename : str, optional
        name of the database file included with the raw data. 
        The default is 'DatasetInformation.xlsx'.

    Returns
    -------
    None.
    
    The function collects raw images across all subfolders in the provided folder.
    It removes duplicated images, downscale and pad the images, run the segmentation
    algorithm and save the final 3d stacks after segmentation has been used to select 
    only the surface of the fly in the original z-stack

    '''
    
    tqdm.pandas()
    # unify databases across folders:
    raw_data_df = create_raw_images_database(read_folder, database_filename)
    
    if len(raw_data_df) == 0:
        print('No dataset information files with the given name have been found')
        return
    
    # clean the destination directory:
    if only_on_new_files == False:
        for f in os.listdir(destination_folder):
            os.remove(os.path.join(destination_folder, f))
    
    # Look for the database file of the preprocessed images to check which 
    # images have been already processed:
        
    try: 
        DatasetInfoPreproc = pd.read_excel(os.path.join(destination_folder,database_filename))
    except:
        DatasetInfoPreproc = pd.DataFrame(columns = ["experiment", "filename_gfp", "filename_dsred", "filename_tl"])
        
    # run row by row over database of files, downsample, segment and save the images,
    # return filenames to create new DatasetInformation.xlsx file in the destination folder
   
    print("Preprocessing of raw images in progress:")

    new_columns = ["experiment", "filename_gfp", "filename_dsred", "filename_tl"]
    raw_data_df[new_columns] = raw_data_df.progress_apply(lambda row: \
    preprocess_and_save(row["image file name"], row["folder"], binning, bit_depth, destination_folder, DatasetInfoPreproc), axis=1)
    
    #raw_data_df = raw_data_df.explode('image file name')
    raw_data_df = raw_data_df[raw_data_df['experiment'].notna()]
    raw_data_df.to_excel(os.path.join(destination_folder,'DatasetInformation.xlsx'))

    return

def create_raw_images_database(root_folder, database_filename = 'DatasetInformation.xlsx'):
    raw_data_df = pd.DataFrame()
    for root, subdirectories, files in os.walk(root_folder):
        filename = os.path.join(root,database_filename)
        try:
            DatasetInfo = pd.read_excel(filename)
            DatasetInfo['folder'] = str(os.path.join(root,''))
            raw_data_df = raw_data_df.append(DatasetInfo)
        except:
            pass

    raw_data_df['File_exists'] = raw_data_df.apply(lambda row: \
    pd.Series(os.path.isfile(os.path.join(row['folder'], 'C1-'+row['image file name']))), axis=1)


    raw_data_df = raw_data_df[raw_data_df['File_exists']]
    
    # Remove duplicated filenames:
    raw_data_df = raw_data_df.drop_duplicates(subset='image file name', keep="last")
    
    return raw_data_df


def preprocess_and_save(image_file_name, folder, binning, bit_depth, destination_folder, DatasetInfoPreproc):
    filename_GFP = os.path.join(folder,'C1-'+image_file_name)
    filename_DsRed = os.path.join(folder,'C2-'+image_file_name)
    filename_TL = os.path.join(folder,'C3-'+image_file_name)
    if os.path.splitext(image_file_name)[0] in DatasetInfoPreproc['experiment'].values:
        return pd.Series([os.path.splitext(image_file_name)[0], 'Preprocessed_C1-'+image_file_name,'Preprocessed_C2-'+image_file_name, 'Preprocessed_C3-'+image_file_name])
    
    try:
        image_GFP = io.imread(filename_GFP)
        image_DsRed = io.imread(filename_DsRed)
        image_TL = io.imread(filename_TL)
    except:
        print("File not found")
        print(filename_TL)
        image_GFP = float("NaN")
        image_DsRed = float("NaN")
        image_TL = float("NaN")
        return pd.Series([float("NaN"), float("NaN"), float("NaN"), float("NaN")])
    
    max_value = 2**bit_depth-1
  
    # rescale images to 16bits:
    image_DsRed = image_DsRed*65536/max_value
    image_GFP = image_GFP*65536/max_value
    image_TL = image_TL*65536/max_value
    
    # Binning:
    new_image_shape = [int(image_DsRed.shape[i]/binning[i]) for i in range(3)]

    image_downscaled = transform.resize(image_GFP, new_image_shape, preserve_range = True)
    image_DsRed_downscaled = transform.resize(image_DsRed, new_image_shape, preserve_range = True)
    image_TL_downscaled = transform.resize(image_TL, new_image_shape, preserve_range = True)

#    image_downscaled = transform.downscale_local_mean(image_GFP, binning)[1:-2,1:-2,1:-2]
#    image_DsRed_downscaled = block_reduce(image_DsRed, binning, np.max, cval=0)[1:-2,1:-2,1:-2]
#    image_TL_downscaled = block_reduce(image_TL, binning, np.max, cval=0)[1:-2,1:-2,1:-2]

    # Segmentation:
    thresholded = segmentation_with_optimized_thresh(image_downscaled)

    # Padding:
    image_downscaled = image_padding(image_downscaled)
    image_DsRed_downscaled = image_padding(image_DsRed_downscaled)
    image_TL_downscaled = image_padding(image_TL_downscaled)
    thresholded  = image_padding(thresholded)
    
    # Clean up the segmentation with morphological transformations:
    thresholded = clean_up_segmented_image(thresholded)
    
    segmented_image_GFP = (image_downscaled)*thresholded
    thresholded = segmented_image_GFP>0
    segmented_image_DsRed = (image_DsRed_downscaled+1)*thresholded
    segmented_image_TL = (image_TL_downscaled+1)*thresholded
    
    image_file_names = [os.path.basename(filename_GFP), os.path.basename(filename_DsRed), os.path.basename(filename_TL)]
    preprocessed_images = [segmented_image_GFP, segmented_image_DsRed, segmented_image_TL ]
    new_file_names = aux_save_images(preprocessed_images, image_file_names, destination_folder)
    
    return pd.Series([os.path.splitext(image_file_name)[0], new_file_names[0], new_file_names[1], new_file_names[2]])

def aux_save_images(images,names,folder):
    file_names_list = list()
    if isinstance(images, list):
        for count, image in enumerate(images):
            filename = 'Preprocessed_'+names[count]
            imsave(os.path.join(folder,filename), image)
            file_names_list.append(filename)
    else:
        filename = names
        imsave(os.path.join(folder,filename), image)
        file_names_list.append(filename)
    return list(file_names_list)


def segmentation_with_optimized_thresh(image, threshold = 1.05, max_iter = 200, fraction_range = [0.025, 0.040]):
    test_thresholded = image > threshold*np.mean(image)
    segm_fract = np.sum(test_thresholded)/test_thresholded.size
    step = 0.01
    count_iter = 1
    delta_segmented_fraction = 0
    while ((segm_fract > fraction_range[1]) or (segm_fract < fraction_range[0])) and count_iter < max_iter:
        if segm_fract > fraction_range[1]:
            count_iter += 1
            threshold += step
            test_thresholded = image > threshold*np.mean(image)
            segm_fract = np.sum(test_thresholded)/test_thresholded.size
            if segm_fract < fraction_range[0]:
                step = 0.1*step
        else:
            threshold -= step
            count_iter += 1
            test_thresholded = image > threshold*np.mean(image)
            segm_fract = np.sum(test_thresholded)/test_thresholded.size
            if segm_fract > fraction_range[1]:
                step = 0.1*step
    #print(count_iter)
    #print(segm_fract)
    return test_thresholded

def image_padding(image, padding = 20):
    # Padding:
    shape = np.asarray(np.shape(image))
    padded_image = np.zeros(shape+2*padding)
    padded_image[padding:-padding,padding:-padding,padding:-padding] = image
    return padded_image

def clean_up_segmented_image(binary_image, closing_r1 = 4, closing_r2 = 8):
    
    filled = morphology.closing(binary_image, morphology.ball(closing_r1))

    label_image = label(filled)
    rp = regionprops(label_image)
    size = max([i.area for i in rp])

    biggest_object = morphology.remove_small_objects(label_image, min_size=size/10)>0

    # Skeletonize the segmented image, slice by slice:
    skeletonized = skeletonize_on_slices(biggest_object)
    
    # Create a pcd from the skeletonized image
    skeleton_pcd, skeleton_values = image_to_pcd(skeletonized)
    
    # Remove outliers from the skeletonized pcd, based on n_neighbrours within a radius:
    uni_down_pcd = skeleton_pcd.uniform_down_sample(every_k_points=3)
    cleaned_pcd, ind = uni_down_pcd.remove_radius_outlier(nb_points=4, radius=3)
    cleaned_pcd, ind = cleaned_pcd.remove_radius_outlier(nb_points=4, radius=3)
    
    # downsampling
    cleaned_pcd = cleaned_pcd.voxel_down_sample(voxel_size=12)

    # Create a mesh that fits through the cleaned points to fill potential holes:
    cleaned_pcd.estimate_normals()
    cleaned_pcd.orient_normals_consistent_tangent_plane(k=30)
    
    radii = [60]
    ball_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(cleaned_pcd, o3d.utility.DoubleVector(radii))
    bbox = cleaned_pcd.get_axis_aligned_bounding_box()
    ball_mesh_crop = ball_mesh.crop(bbox)

    # Resample the mesh and create the final point cloud:
    resampled_pcd = ball_mesh_crop.sample_points_uniformly(number_of_points = 5000)
    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points = o3d.utility.Vector3dVector( np.concatenate((cleaned_pcd.points, resampled_pcd.points), axis=0) )


    final_pcd_values = np.ones(np.asarray(final_pcd.points).shape[0])
    final_image = pcd_to_image(final_pcd, final_pcd_values, binary_image.shape)
    final_image = morphology.dilation(final_image, morphology.ball(10))

    return final_image

def skeletonize_on_slices(image_3d):
    
    result = np.zeros(image_3d.shape)
    
    for i in range(image_3d.shape[1]):
        image = image_3d[:,i,:]
        skeleton = morphology.skeletonize(image)
        result[:,i,:] = skeleton

    return result 



if __name__ == '__main__':
    
    ## %matplotlib qt  ##
    read_data_folder = "../../data_2/01_raw"
    destination_folder = "../../data_2/02_preprocessed"
    preprocess_and_segment_images(read_data_folder, destination_folder, binning = [2,2,2] , bit_depth = 12)
    #database_name = "DatasetInformation_Part1.xlsx"

    
