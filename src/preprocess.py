##################################################
## Preprocessing and segmentation of fla abdomens
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
import napari
from skimage import morphology
from skimage.measure import label, regionprops, block_reduce
from scipy import stats
import matplotlib.pyplot as plt
from tifffile import imsave
from tqdm import tqdm



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
            
    try: 
        DatasetInfoPreproc = pd.read_excel(os.path.join(destination_folder,database_filename))
    except:
        DatasetInfoPreproc = pd.DataFrame(columns = ["experiment", "filename_fl", "filename_tl"])
        
    # run row by row over database of files, downsample, segment and save the images,
    # return filenames to create new DatasetInformation.xlsx file in the destination folder
   
    print("Preprocessing of raw images in progress:")

    new_columns = ["experiment", "filename_fl", "filename_tl"]
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

    #Drop images not used for the analysis:
    #raw_data_df = raw_data_df.drop(raw_data_df[raw_data_df["used for analysis"]=="no"].index)

    # Remove duplicated filenames:
    raw_data_df = raw_data_df.drop_duplicates(subset='image file name', keep="last")
    
    return raw_data_df


def preprocess_and_save(image_file_name, folder, binning, bit_depth, destination_folder, DatasetInfoPreproc):

    filename_fl = os.path.join(folder,'C1-'+image_file_name)
    filename_dic = os.path.join(folder,'C2-'+image_file_name)
    
    if os.path.splitext(image_file_name)[0] in DatasetInfoPreproc['experiment'].values:
        return pd.Series([os.path.splitext(image_file_name)[0], 'Preprocessed_C1-'+image_file_name,'Preprocessed_C2-'+image_file_name])
    try:
        image_fl = io.imread(filename_fl)
        image_dic = io.imread(filename_dic)
    except:
        image_fl = float("NaN")
        image_dic = float("NaN")
        return pd.Series([float("NaN"), float("NaN"), float("NaN")])
    
    max_value = 2**bit_depth-1
    
    # inverting dic image:
    image_dic = max_value - image_dic
    
    # rescale images to 16bits:
    image_dic = image_dic*65536/max_value
    image_fl = image_fl*65536/max_value

    # Binning:
    image_downscaled = transform.downscale_local_mean(image_fl, binning)[1:-2,1:-2,1:-2]
    image_dic_downscaled = block_reduce(image_dic, binning, np.max, cval=0)[1:-2,1:-2,1:-2]

    # Segmentation:
    thresholded = segmentation_with_optimized_thresh(image_downscaled)

    # Padding:
    image_downscaled = image_padding(image_downscaled)
    image_dic_downscaled = image_padding(image_dic_downscaled)
    thresholded  = image_padding(thresholded)
    
    # Clean up the segmentation with morphological transformations:
    thresholded = clean_up_segmented_image(thresholded)
    
    segmented_image_fl = (image_downscaled+1)*thresholded
    segmented_image_dic = (image_dic_downscaled+1)*thresholded

    image_file_names = [os.path.basename(filename_fl), os.path.basename(filename_dic)]
    preprocessed_images = [segmented_image_fl, segmented_image_dic]
    new_file_names = aux_save_images(preprocessed_images, image_file_names, destination_folder)
    
    return pd.Series([os.path.splitext(image_file_name)[0], new_file_names[0], new_file_names[1]])

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


def segmentation_with_optimized_thresh(image, threshold = 1.05, max_iter = 200, fraction_range = [0.04, 0.08]):
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
                
    return test_thresholded

def image_padding(image, padding = 20):
    # Padding:
    shape = np.asarray(np.shape(image))
    padded_image = np.zeros(shape+2*padding)
    padded_image[padding:-padding,padding:-padding,padding:-padding] = image
    return padded_image

def clean_up_segmented_image(binary_image, dilation_r = 2, closing_r1 = 4, closing_r2 = 8):
    dilated = morphology.dilation(binary_image, morphology.ball(dilation_r))

    # dilate and erode again to fill small holes:
    filled = morphology.closing(dilated, morphology.ball(closing_r1))

    label_image = label(filled)
    rp = regionprops(label_image)
    size = max([i.area for i in rp])

    biggest_object = morphology.remove_small_objects(label_image, min_size=size-1)>0
    biggest_object = morphology.erosion(biggest_object, morphology.ball(dilation_r))
    
    # fill small holes in the final segmented image:
    biggest_object = morphology.closing(biggest_object, morphology.ball(closing_r2))
    
    return biggest_object



if __name__ == '__main__':
    
    ## %matplotlib qt  ##
    destination_folder = "/media/ceolin/Data/Lab Gompel/Projects/Fly_Abdomens/data/02_preprocessed"
    read_folder = "/media/ceolin/Data/Lab Gompel/Projects/Fly_Abdomens/data/01_raw"
    
    preprocess_and_segment_images(read_folder, destination_folder)
    #database_name = "DatasetInformation_Part1.xlsx"

    
