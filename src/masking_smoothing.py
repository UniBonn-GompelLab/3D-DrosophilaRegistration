##################################################
## Preprocessing of registered wing images 
##################################################
## Author: Stefano
## Version: March 2022
##################################################

import pandas as pd
import numpy as np
import os
from PIL import Image
from scipy import signal
from numpy.fft  import fft2, ifft2, ifftshift
from skimage.measure import block_reduce
from tqdm import tqdm

def preprocess_registered_images(
    read_folder, destination_folder, database_registered, database_info, \
    mask_filename, smooth_x, smooth_y, bcg_construct = None, binning=None):
    '''
    Parameters
    ----------
    read_folder : str
        path of the folder containing the raw data.
    destination_folder : str
        path of the folder where preprocessed images will be saved.
    mask_filename : str
        full path to the mask file.
    smoothing_sigma : int
        radius of gaussian smoothing to apply to the bcg images, in pixels.
    bcg_construct : str, optional
        name of the fly stock to use as background. The default is None.
    database_registered : str, optional
        name of the database file included with the raw data. 
        The default is 'DatasetInfo.csv'.

    Returns
    -------
    None.
    
    The function collects raw images across all subfolders in the provided folder.
    It removes duplicated images, applies the mask to all the images and removes 
    the average background. Then it saves the images in the destination folder, together
    with a new 'DatasetInfo.csv' file to open the images again.

    '''
    
    
    
    # unify databases across folders:
    raw_data_df = create_raw_images_database(read_folder, database_registered, database_info)
    
    # clean the destination directory:
    for f in os.listdir(destination_folder):
        os.remove(os.path.join(destination_folder, f))
        
    # create the mask:
    mask = np.asarray(Image.open(mask_filename))/255
    #mask = create_common_mask(raw_data_df, mask)
    
    # create a background
    if bcg_construct:
        print("Skipping background removal - Not implemented")
        #bcg = create_background(raw_data_df, mask, smoothing_sigma, bcg_construct)
    else:
        bcg = None
        
    # Initialize variables for smoothing:
    fft_gauss_kernel = fft_gaussian_smoothing_kernel(mask, smooth_x, smooth_y)
    normalization_mask = (1/(aux_convolution(mask,fft_gauss_kernel)*mask+(mask==0)))*mask

    # run row by row, apply the mask, remove the background and save the results
    # return filenames to create new DatasetInformation.csv file in the destination folder
    print("Preprocessing of registered raw images in progress:")
    tqdm.pandas()
    raw_data_df["file name"] = raw_data_df.progress_apply(lambda row: \
    preprocess_and_save(row["full path"], mask, fft_gauss_kernel, normalization_mask, bcg, destination_folder, binning), axis=1)
    
    raw_data_df['folder'] = str(os.path.join(destination_folder,''))
    raw_data_df.to_csv(os.path.join(destination_folder,'DatasetInfo.csv'))
    
    # save common mask:
    mask = Image.fromarray(mask.astype(np.uint8))
    mask = mask.convert("I;16")
    mask.save(os.path.join(destination_folder,"mask.tif"))

    return


def create_raw_images_database(folder, database_registered_path, database_info_path):
    
    raw_data_df = pd.read_csv(database_registered_path)
    info_df = pd.read_excel(database_info_path)
    info_df = info_df[["image file name", "construct"]]
    info_df = info_df.rename(columns={"image file name": "original file name"})
    
    raw_data_df['folder'] = str(os.path.join(folder,''))
    
    for index, row in raw_data_df.iterrows():
        raw_data_df.loc[index, 'full path'] = os.path.join(row['folder'], row['file name'])
        raw_data_df.loc[index, 'original file name'] = row['file name'][13:]
    
    raw_data_df = pd.merge(raw_data_df, info_df, on="original file name", how="left")
    
    return raw_data_df

def create_common_mask(raw_data_df, mask):

    image_names = raw_data_df["file name"].values
    
    for image_file_name in image_names:
        image = Image.open(image_file_name)
        image = np.asarray(image)
        mask = mask*(image>0)
        
    return mask

def preprocess_and_save(image_file_name, mask, fft_gauss_kernel, normalization_mask, bcg, destination_folder, binning):
    
    image = Image.open(image_file_name)
    image = np.asarray(image)
    image = aux_apply_mask(mask, image)

    if bcg:
        image = image-bcg
        image *= (image>0)

    image = smoothing(image, fft_gauss_kernel, normalization_mask)

    if binning:
        image = aux_downsampling(image, binning)
        
    image_file_name = os.path.basename(image_file_name)
    new_file_name = aux_save_image(image, image_file_name, destination_folder)

    return new_file_name

def fft_gaussian_smoothing_kernel(mask, smoothing_sigma_h, smoothing_sigma_v):
    # Definition of the Gaussian kernel for blurring the background
    kernlen = mask.shape[1]
    gkern1d_h = signal.gaussian(kernlen, std=smoothing_sigma_h).reshape(kernlen, 1)
    kernlen = mask.shape[0]
    gkern1d_v = signal.gaussian(kernlen, std=smoothing_sigma_v).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d_v, gkern1d_h)
    gkern2d = gkern2d/sum(map(sum, gkern2d))
    fft_kernel = fft2(ifftshift(gkern2d))   
    return fft_kernel


def smoothing(image, fft_kernel, normalized_mask):
    result = aux_convolution(image, fft_kernel)*normalized_mask
    return result

def aux_apply_mask(mask, image):
    # The mask is an image where False = 0 and True = 255, hence the normalization
    # A small offset is used to avoid 0 pixels inside the mask.
    result =  (mask)*(image+0.0001)
    return result

def aux_convolution(image, fft_kernel):
    result = np.real(ifft2(fft2(image)*fft_kernel))
    return result

def aux_downsampling(image, binning):
    result = block_reduce(image,block_size=(binning,binning),func=np.mean)
    return result

def aux_save_image(image, filename, folder):
    im = Image.fromarray(image.astype(np.uint16))
    im = im.convert("I;16")
    im.save(os.path.join(folder,filename))
    return filename



if __name__ == '__main__':
    
    ## %matplotlib qt  ##
    destination_folder = "../../data/07_masked_and_smooth"
    read_folder = "../../data/06_warped"
    mask = '../../data/References_and_masks/2D_mask_abdomen_2.tif'
    database_registered_images = "../../data/06_warped/dataframe_info.csv"
    database_info = "../../data/04_projected/DatasetInformation.xlsx"
    
    
    preprocess_registered_images(read_folder, destination_folder, database_registered_images, database_info, mask_filename = mask, smoothing_sigma=4, binning=1)