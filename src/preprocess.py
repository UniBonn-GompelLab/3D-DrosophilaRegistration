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
    read_folder, destination_folder, downscaling=(2,2,2), bit_depth=8, only_on_new_files = True,\
    database_filename = 'DatasetInformation.xlsx'):
    
    '''
    Parameters
    ----------
    read_folder : str
        path of the folder containing the raw data.
    destination_folder : str
        path of the folder where preprocessed images will be saved.
    downscaling: (float, float, float)
        downscaling factor along z, x, y, in pixels.
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
    preprocess_and_save(row["image file name"], row["folder"], downscaling, bit_depth, destination_folder, DatasetInfoPreproc), axis=1)
    
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


def preprocess_and_save(image_file_name, folder, downscaling, bit_depth, destination_folder, DatasetInfoPreproc):
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

    # rescale images to 16bits:    
    max_value = 2**bit_depth-1
  
    image_DsRed = image_DsRed*65536/max_value
    image_GFP = image_GFP*65536/max_value
    image_TL = image_TL*65536/max_value
    
    # Resizing of the images:
    new_image_shape = [int(image_DsRed.shape[i]/downscaling[i]) for i in range(3)]
    image_GFP_downscaled = transform.resize(image_GFP, new_image_shape, preserve_range = True)
    image_DsRed_downscaled = transform.resize(image_DsRed, new_image_shape, preserve_range = True)
    image_TL_downscaled = transform.resize(image_TL, new_image_shape, preserve_range = True)

    # Segmentation:
    thresholded = segmentation_with_optimized_thresh(image_GFP_downscaled)
    
    # Make sure there are no zero pixels in the original image:
    image_GFP_downscaled = image_GFP_downscaled
    image_DsRed_downscaled = image_DsRed_downscaled
    image_TL_downscaled = image_TL_downscaled
        
    # Padding:
    image_GFP_downscaled    = image_padding(image_GFP_downscaled)
    image_DsRed_downscaled  = image_padding(image_DsRed_downscaled)
    image_TL_downscaled     = image_padding(image_TL_downscaled)
    thresholded             = image_padding(thresholded)
    
    # Clean up the segmentation with morphological transformations:
    thresholded = clean_up_segmented_image(thresholded, image_GFP_downscaled)
    
    # Apply the mask (thresholded image) to all channels:
    segmented_image_GFP = (image_GFP_downscaled)*thresholded
    segmented_image_DsRed = (image_DsRed_downscaled)*thresholded
    segmented_image_TL = (image_TL_downscaled)*thresholded
    
    # Save the segmented images:
    image_file_names = [os.path.basename(filename_GFP), os.path.basename(filename_DsRed), os.path.basename(filename_TL)]
    preprocessed_images = [segmented_image_GFP, segmented_image_DsRed, segmented_image_TL]
    new_file_names = aux_save_images(preprocessed_images, image_file_names, "Preprocessed_", destination_folder)
    
    return pd.Series([os.path.splitext(image_file_name)[0], new_file_names[0], new_file_names[1], new_file_names[2]])

def aux_save_images(images, names, prefix, folder):
    """
    Save a list of images in a folder adding a common prefix to the filenames.

    Parameters
    ----------
    images : TYPE
        DESCRIPTION.
    names : TYPE
        DESCRIPTION.
    prefix : TYPE
        DESCRIPTION.
    folder : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
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


def segmentation_with_optimized_thresh(image, threshold = 1.05, max_iter = 200, fraction_range = [0.04, 0.05]):
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
        a binary mask of the segmented volume.

    """
    thresholded_image = image > threshold*np.mean(image)
    
    segm_fract = np.sum(thresholded_image)/thresholded_image.size
    step = 0.01
    
    n_iterations = 1
    delta_segmented_fraction = 0
    
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

def image_padding(image, padding = 20, cval = 0):
    """
    Padding of a 3d image with a constant value
    
    Parameters
    ----------
    image : 3d numpy array
        3d image.
    padding : int, optional
        extent of the padding. The default is 20.
    cval: int, optional
        constant value used for padding the image. The default is 0. 

    Returns
    -------
    padded_image : TYPE
        DESCRIPTION.

    """
    shape = np.asarray(np.shape(image))
    padded_image = np.ones(shape+2*padding)*cval
    padded_image[padding:-padding,padding:-padding,padding:-padding] = image
    return padded_image

def clean_up_segmented_image(binary_image, image, closing_r = 4, dilation_r = 8, mesh_radius = 30):
    """
    This function refines the segmentation of a surface in a 3d image using 
    morphological transformations (closing, dilation), selecting local maxima 
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
    None.

    """
    
    filled = morphology.closing(binary_image, morphology.ball(closing_r))

    label_image = label(filled)
    rp = regionprops(label_image)
    size = max([i.area for i in rp])

    biggest_objects_mask = morphology.remove_small_objects(label_image, min_size=size/10)>0

    # Create a max that selects the local maxima along the z direction:
    thresholded_image = local_maxima_z( biggest_objects_mask * image, mesh_radius)

    # To fill potential holes in the segmented object we create a point cloud object
    # from the mask, and fit a mesh on the points.
    
    pcd, pcd_values = image_to_pcd(thresholded_image)
    
    # Remove isolated points from the points cloud, based on the number of 
    # neighbours within a given radius:
        
    uni_down_pcd = pcd.uniform_down_sample(every_k_points=3)
    cleaned_pcd, ind = uni_down_pcd.remove_radius_outlier(nb_points=4, radius=closing_r)
    cleaned_pcd, ind =  cleaned_pcd.remove_radius_outlier(nb_points=4, radius=closing_r)
    
    # Downsampling the point cloud to make the sampling more uniform:
    cleaned_pcd = cleaned_pcd.voxel_down_sample(voxel_size=5)

    # Create a mesh that fits through the cleaned points to fill potential holes:
    cleaned_pcd.estimate_normals()
    cleaned_pcd.orient_normals_consistent_tangent_plane(k=30)
    normals = -np.asarray(cleaned_pcd.normals)
    cleaned_pcd.normals = o3d.utility.Vector3dVector(normals)  
    ball_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(cleaned_pcd, o3d.utility.DoubleVector([mesh_radius]))
    bbox = cleaned_pcd.get_axis_aligned_bounding_box()
    ball_mesh_crop = ball_mesh.crop(bbox)

    # Resample the mesh and create the final point cloud:
    resampled_pcd = ball_mesh_crop.sample_points_uniformly(number_of_points = 50000)
    final_pcd = o3d.geometry.PointCloud()
    
    final_pcd.points = o3d.utility.Vector3dVector( np.concatenate((pcd.points, resampled_pcd.points), axis=0) )
    final_pcd_values = np.ones(np.asarray(final_pcd.points).shape[0])

    # Convert the point cloud into a binary image and apply a dilation:
    final_image = pcd_to_image(final_pcd, final_pcd_values, binary_image.shape)
    final_image = morphology.dilation(final_image, morphology.ball(dilation_r))

    return final_image

from scipy.signal import find_peaks
def local_maxima_z(image, dist):
    """
    This function create a mask of the image which select the local maxima along 
    the z direction which are separated by a minimum distance,
    
    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    dist : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # maxima along z
    mask_local_maxima_z = np.zeros(image.shape)
    
    for i in range(image.shape[1]):
        for j in range(image.shape[2]):
            peaks, _ = find_peaks(image[:,i,j], distance = dist)
            for p in peaks:
                mask_local_maxima_z[p,i,j] = 1
    
    return mask_local_maxima_z



if __name__ == '__main__':
    
    ## %matplotlib qt  ##
    read_data_folder = "../../data_2/01_raw"
    destination_folder = "../../data_2/02_preprocessed"
    preprocess_and_segment_images(read_data_folder, destination_folder, downscaling = [1,2.5,2.5] , bit_depth = 12)


    
