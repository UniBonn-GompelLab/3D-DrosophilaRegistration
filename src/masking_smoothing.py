#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Function(s) to smooth and apply a mask to a series of 2D images.

@author: ceolin
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from scipy import signal
from numpy.fft import fft2, ifft2, ifftshift
from skimage.measure import block_reduce
from tqdm import tqdm


def run_smoothing_masking(
        read_folder, destination_folder, df_warped, df_info,
        mask_filename, smooth_x, smooth_y, bcg_types=None, bcg_channels=None, refine_mask = True, binning=None):
    '''
    This function merges the information about the images in the two dataframes,
    calculates a common mask for all the images for each channel by refining the
    provided binary mask. Optionally it also calculates a background image for
    a selection of channels using images of the given types. It removes the background
    from the images, smooths, masks them and saves the final result.

    Parameters
    ----------
    read_folder : str
        path of the folder containing the images.
    destination_folder : str
        path of the folder where smoothed images will be saved.
    df_warped : pandas dataframe
        DESCRIPTION.
    df_info : pandas dataframe
        DESCRIPTION.
    mask_filename : str
        full path to the mask file.
    smooth_x : int
        width of gaussian smoothing to apply along the x direction.
    smooth_y : int
        width of gaussian smoothing to apply along the y direction.
    bcg_types : list of str, optional
        image type(s) to use as background. The default is None.
    bcg_channels : list of str, optional
        channels for which bcg correction has to be applied. The default is None.
    binning : int, optional
        binning factor to apply to the final images. The default is None.

    Returns
    -------
    None.

    '''

    df_warped = merge_image_information(read_folder, df_warped, df_info)

    # clean the destination directory:
    for f in os.listdir(destination_folder):
        os.remove(os.path.join(destination_folder, f))

    # create a mask for each channel:
    mask = np.asarray(Image.open(mask_filename))
    mask = mask / np.max(mask)
    if refine_mask:
        channel_masks = create_common_mask_for_each_channel(df_warped, mask)
    else:
        channel_masks = {}
        for channel in df_warped["channel"].unique():
            channel_masks[channel] = mask

    # Initialize variables for fast smoothing:
    fft_gauss_kernel = fft_gaussian_smoothing_kernel(
        mask.shape, smooth_x, smooth_y)

    normalization_masks = {}
    for channel in df_warped["channel"].unique():
        norm_mask_tmp = (1 / aux_convolution(
            channel_masks[channel], fft_gauss_kernel)*mask+(mask == 0))*channel_masks[channel]
        normalization_masks[channel] = np.nan_to_num(norm_mask_tmp)

    # create a dictionary with a background image for each channel in the dataset:
    if bcg_types:
        bcg = create_backgrounds(df_warped, bcg_types, bcg_channels, channel_masks,
                                 fft_gauss_kernel, normalization_masks)
    else:
        bcg = None

    # for each image file in the dataframe open the image, apply the mask,
    # remove the background and save the final image in the destination folder.
    # Update filenames in the dataframe and create new .csv file in the
    # destination folder.

    print("Smoothing and masking of the registered images in progress:")

    tqdm.pandas()
    df_warped["file name"] = df_warped.progress_apply(
        lambda row: mask_smooth_and_save(
            row["full path"], row["channel"], channel_masks, fft_gauss_kernel,
            normalization_masks, bcg, destination_folder, binning),
        axis=1)

    df_warped['folder'] = str(os.path.join(destination_folder, ''))
    df_warped.drop(columns=['full path'], inplace = True)
    df_warped.to_csv(os.path.join(destination_folder, 'DatasetInfo.csv'),  index=False)

    # save common masks and background images for documentation:
    for channel in df_warped["channel"].unique():

        mask = Image.fromarray(channel_masks[channel].astype(np.uint8))
        mask = mask.convert("I;16")
        mask.save(os.path.join(destination_folder, "mask_"+channel+".tif"))
        
        if bcg:
            try:
                background = Image.fromarray(bcg[channel].astype(np.uint16))
                background = background.convert("I;16")
                background.save(os.path.join(
                    destination_folder, "bcg_"+channel+".tif"))

            except KeyError:
                pass

    return


def merge_image_information(folder, df_file_paths, df_image_info):
    """
    Parameters
    ----------
    folder : str
        path to the folder containing the warped images.
    df_file_paths : pandas dataframe
        dataframe containing the file names of the warped images.
    df_image_info : pandas dataframe
        dataframe containing the additional info about the images, like genotype, etc.

    Returns
    -------
    df_merged : pandas dataframe

    """

    df_image_info = df_image_info[["image file name", "type"]]
    df_image_info = df_image_info.rename(
        columns={"image file name": "original file name"})

    df_file_paths['folder'] = str(os.path.join(folder, ''))

    for index, row in df_file_paths.iterrows():
        df_file_paths.loc[index, 'full path'] = os.path.join(
            row['folder'], row['file name'])
        df_file_paths.loc[index, 'original file name'] = row['file name'][13:]

    df_merged = pd.merge(df_file_paths, df_image_info,
                         on="original file name", how="left")

    return df_merged


def create_common_mask_for_each_channel(df_images, mask):
    """
    This function calculates a separate common mask for the images of each channel
    in the provided dataframe.

    Parameters
    ----------
    df_images : pandas dataframe
        dataframe containing the 'full path' to the images and their associated
        'channel'.
    mask : numpy 2D array
        common base binary mask.

    Returns
    -------
    channel_masks : dict

    """
    channel_masks = {}
    channel_names = df_images["channel"].unique()
    gauss_kernel_fft = fft_gaussian_smoothing_kernel(mask.shape, 1, 1)
    
    for channel in channel_names:

        df_channel = df_images[df_images["channel"] == channel]
        channel_image_paths = df_channel["full path"].values
        tmp_channel_mask = create_common_mask(channel_image_paths, mask)
        # fill small holes in channel mask:
        tmp_channel_mask = aux_convolution(tmp_channel_mask, gauss_kernel_fft )  > 0.5
        channel_masks[channel] = tmp_channel_mask

    return channel_masks


def create_common_mask(list_image_files, mask):
    """
    This function opens all the images in the list of image files, compares them
    with the mask image and refines the mask to create a common mask in case some
    images cover a smaller surface compared to the provided mask.

    Parameters
    ----------
    list_image_files : list of str
        list of paths to image files.
    mask : numpy 2D array
        binary mask.

    Returns
    -------
    mask : numpy 2D array
        refined binary mask.

    """
    for image_file_name in list_image_files:
        image = Image.open(image_file_name)
        image = np.asarray(image)
        mask = mask*(image > 0)

    return mask


def mask_smooth_and_save(image_file_name, channel, masks_dict, smooth_kernel_fft, 
                         normalization_masks, bcgs_dict, destination_folder, binning):
    """
    This function opens an image, applies a mask, smooths the image using the
    provided fourier transform of the smoothing kernel, normalizes the smooth
    image to compensate for the leaking of the signal outside the masked region
    and saves the final result in the destination folder.
    
    Parameters
    ----------
    image_file_name : str
        path to the image.
    channel : str
        channel of the image.
    masks_dict : dict
        dictionary containing binary masks for the different channels as numpy
        2D arrays.
    smooth_kernel_fft : numpy 2D array
        fourier transform of smoothing kernel.
    normalization_masks : dict
        dictionary containing normalization for the smoothing for the different
        channels as numpy 2D arrays.
    bcgs_dict : dict
        dictionary containing average background images for the different
        channels as numpy 2D arrays.
    destination_folder : str
        path of the directory where the processed image will be saved.
    binning : int
        binning factor for the final image.

    Returns
    -------
    new_file_name : str

    """

    image = Image.open(image_file_name)
    image = np.asarray(image)
    image = aux_apply_mask(masks_dict[channel], image)

    if bcgs_dict:
        try:
            image = image - bcgs_dict[channel]
            image *= (image > 0)  # set negative values to zero
        except KeyError:
            pass

    image = smooth_normalize(image, smooth_kernel_fft,
                             normalization_masks[channel])

    if binning:
        image = aux_downsampling(image, binning)

    image_file_name = os.path.basename(image_file_name)
    aux_save_image_16bits(image, image_file_name, destination_folder)
    new_file_name = image_file_name

    return new_file_name


def create_backgrounds(df_images, bcg_types, bcg_channels, masks_dict, smooth_kernel_fft, normalization_masks):
    """
    This function calculates an average background image for each specified channel
    using images of the selected types. It returns a dictionary that contains the
    average background for each channel.

    Parameters
    ----------
    df_images : pandas dataframe
        DESCRIPTION.
    bcg_types : list of str
        names of the types of images to use as background.
    bcg_channels : list of str
        names of the channels for which to calculate the background.
    masks_dict : dict
        dictionary containing binary masks for the different channels as numpy
        2D arrays.
    smooth_kernel_fft : numpy 2D array
        fourier transform of smoothing kernel.
    normalization_masks : dict
        dictionary containing normalization for the smoothing for the different
        channels as numpy 2D arrays.
    Returns
    -------
    background_images : dict

    """

    background_images = {}

    for channel in bcg_channels:

        df_bcg = df_images[(df_images["type"].isin(bcg_types)) & 
                           (df_images["channel"] == channel)]
        bcg_image_paths = df_bcg["full path"].values

        temp_bcg_image = np.zeros(normalization_masks[channel].shape)

        for image_file_name in bcg_image_paths:
            image = Image.open(image_file_name)
            image = np.asarray(image)
            temp_bcg_image += image/len(bcg_image_paths)

        temp_bcg_image = smooth_normalize(
            temp_bcg_image, smooth_kernel_fft, normalization_masks[channel])
        background_images[channel] = temp_bcg_image*masks_dict[channel]

    return background_images


def fft_gaussian_smoothing_kernel(kernel_shape, smoothing_sigma_h, smoothing_sigma_v):
    """
    This function calculates the fast Fourier transform of a 2D gaussian smoothing
    kernel.

    Parameters
    ----------
    kernel_shape : list
        shape of the kernel.
    smoothing_sigma_h : float
        width of the gaussian kernel along axis 1.
    smoothing_sigma_v : float
        width of the gaussian kernel along axis 0.

    Returns
    -------
    fft_kernel : numpy 2D array
        fourier transform of the gaussian kernel.

    """
    kernlen = kernel_shape[1]
    gkern1d_h = signal.gaussian(
        kernlen, std=smoothing_sigma_h).reshape(kernlen, 1)
    kernlen = kernel_shape[0]
    gkern1d_v = signal.gaussian(
        kernlen, std=smoothing_sigma_v).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d_v, gkern1d_h)
    gkern2d = gkern2d/sum(map(sum, gkern2d))
    fft_kernel = fft2(ifftshift(gkern2d))
    return fft_kernel


def smooth_normalize(image, fft_kernel, normalized_mask):
    """

    Parameters
    ----------
    image : numpy 2D array
    fft_kernel : numpy 2D array
        fourier transform of smoothing kernel.
    normalized_mask : numpy 2D array
        normalization image to compensate for boundary effects of the smoothing.

    Returns
    -------
    result : numpy 2D array

    """
    result = aux_convolution(image, fft_kernel)*normalized_mask
    return result


def aux_apply_mask(mask, image, rel_offset=0.01):
    '''
    This function applies a binary mask to an image. A small relative offset can
    be used to avoid pixels with a zero value inside the masked region.

    Parameters
    ----------
    mask : numpy 2D array
    image : numpy 2D array
    rel_offset : float, optional
        relative offset. The default is 0.01.

    Returns
    -------
    masked_image  :  numpy 2D array

    '''
    if np.max(mask) != 1:
        mask = mask/np.max(mask)

    absolute_offset = np.min(image[image > 0])*rel_offset
    masked_image = (mask)*(image+absolute_offset)

    return masked_image


def aux_convolution(image, fft_kernel):
    """
    This function calculate a convolution between an image and an image kernel.
    It takes as input the image in the real space and the fourier transform of
    the kernel and returns the result of the convolution in the real space.

    Parameters
    ----------
    image : numpy 2D array
    fft_kernel : numpy 2D array
        Fourier transform of the kernel.

    Returns
    -------
    result : numpy 2D array

    """
    result = np.real(ifft2(fft2(image)*fft_kernel))
    return result


def aux_downsampling(image, binning):
    """
    This function downsample an image of a given integer factor. The brightness
    values of multiple pixels binned together is averaged.

    Parameters
    ----------
    image : numpy 2D array
    binning : int

    Returns
    -------
    result : numpy 2D array

    """
    result = block_reduce(image, block_size=(binning, binning), func=np.mean)
    return result


def aux_save_image_16bits(image, filename, folder):
    """
    This function saves a numpy bidimensional arrays as a grayscale image with
    a depth of 16 bits.

    Parameters
    ----------
    image : numpy 2D array
    filename : str
    folder : str

    Returns
    -------
    None.

    """

    pil_image = Image.fromarray(image.astype(np.uint16))
    pil_image = pil_image.convert("I;16")
    pil_image.save(os.path.join(folder, filename))

    return


if __name__ == '__main__':

    ## %matplotlib qt  ##
    destination_folder = "../../data/07_masked_and_smooth"
    read_folder = "../../data/06_warped"
    mask = '../../data/References_and_masks/mask_2d.tif'
    database_registered_images = "../../data/06_warped/dataframe_info.csv"
    database_info = "../../data/04_projected/DatasetInformation.xlsx"

    df_images = pd.read_csv(database_registered_images)
    df_info = pd.read_excel(database_info)

    run_smoothing_masking(read_folder, destination_folder, df_images, df_info,
                          mask_filename=mask, smooth_x=1, smooth_y=4, bcg_types=['empty'], 
                          bcg_channels=['C2'],  binning=None)
