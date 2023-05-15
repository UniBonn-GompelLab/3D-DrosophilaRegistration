#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function(s) used to perform a PCA analysis and visualize a dataset of masked
2D images.

@author: ceolin
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy.sparse import coo_matrix
from sklearn.decomposition import PCA
import seaborn as sns


def reshape_all_images_for_PCA(df_images, mask_filename):
    """
    This function load all the images listed in the input dataframe and preprocess
    them for the PCA analysis. The preprocessing consists of flattening the
    image in the area selected by a binary mask. The flattened arrays are stored
    directly in the dataframe.

    Parameters
    ----------
    df_images : dataframe
        dataframe containing the file names of the images to analyze.
    mask_filename : str
        full path to a binary mask image.

    Returns
    -------
    df_images : TYPE
        DESCRIPTION.

    """

    mask = np.asarray(Image.open(mask_filename))
    mask = mask/np.max(mask)
    df_images['file name'] = df_images['folder']+df_images['file name']

    print("Reshaping data for PCA")
    example_image = np.asarray(Image.open(df_images.loc[0, 'file name']))
    img_memory = example_image.nbytes
    print("-----")
    print("** Warning:\n Loading all the images will take up to "+str(len(df_images)*img_memory/1024000) +
          " MB. \n Consider binning the images or applying the PCA analysis on averages of each group of images.")
    print("-----")
    
    applied = df_images.apply(lambda row: aux_open_and_reshape_for_PCA(
        row['file name'], mask), axis=1, result_type='expand')
    applied.columns = ['I_reshaped', 'x', 'y']

    df_images = pd.concat([df_images, applied], axis=1)

    return df_images


def aux_open_and_reshape_for_PCA(image_file_name, mask, rel_offset=0.01):
    """
    This function opens a grayscale image, and calls the reshape function to
    flatten the area of the image selected by the binary mask.

    Parameters
    ----------
    image_file_name : str
    mask : 2D numpy array
        binary mask.
    rel_offset : float, optional
        Relative offset applied to the image to avoid pixels with a zero value
        inside the masked area. 0-valued pixels are set to rel_offset * the
        minimum non zero pixel value of the image. The default is 0.01.

    Returns
    -------
    bright : numpy array
        brightness of the image.
    row : nump array
        rows of the pixels.
    col : numpy array
        columns of the pixels.

    """

    image = Image.open(image_file_name)
    image = np.asarray(image)
    bright, row, col = aux_reshape_for_PCA(image, mask,  rel_offset)

    return bright, row, col


def aux_reshape_for_PCA(image, mask, rel_offset=0.01):
    """
    This function flattens the area of the image selected by the binary mask.

    Parameters
    ----------
    image : 2D numpy array
    mask : 2D numpy array
        binary mask.
    rel_offset : float, optional
        Relative offset applied to the image to avoid pixels with a zero value
        inside the masked area. 0-valued pixels are set to rel_offset * the
        minimum non zero pixel value of the image. The default is 0.01.

    Returns
    -------
    bright : numpy array
        brightness of the image.
    row : nump array
        rows of the pixels.
    col : numpy array
        columns of the pixels.

    """

    image = np.nan_to_num(image)
    image_min = np.min(image[np.nonzero(image)])
    image = (mask)*(image+image_min*(rel_offset))
    sparse_image = coo_matrix(image)
    bright = sparse_image.data
    row = sparse_image.row
    col = sparse_image.col

    return bright, row, col


def reshape_averages_for_PCA(df_images, mask_filename, column_for_grouping="type"):
    """
    This function loops through all the images listed in the input dataframe and
    computes an average image for each group. Then it preprocesses the averages
    for the PCA analysis. The preprocessing consists of flattening the image in
    the area selected by a binary mask. The average and flattened arrays are stored
    in a new dataframe.

    Parameters
    ----------
    df_images : dataframe
        dataframe containing the file names of the images to analyze.
    mask_filename : str
        full path to a binary mask image.
    column_for_grouping: str

    Returns
    -------
    df_averages : dataframe

    """

    mask = np.asarray(Image.open(mask_filename))
    df_images['file name'] = df_images['folder']+df_images['file name']

    averages_list = []
    average_reshaped_list = []
    average_x_list = []
    average_y_list = []
    n_replicates = []
    groups = df_images[column_for_grouping].unique()

    for group in groups:

        image_names = df_images.loc[df_images[column_for_grouping]
                                    == group, "file name"].values

        if len(image_names) > 0:

            temp_image = Image.open(image_names[0])
            average_img = np.asarray(temp_image)/len(image_names)

            for image_file_name in image_names[1:]:
                temp_image = Image.open(image_file_name)
                temp_image = np.asarray(temp_image)/len(image_names)
                average_img = (average_img + temp_image)

        average_reshaped, average_x, average_y = aux_reshape_for_PCA(
            average_img, mask)

        averages_list.append(average_img)
        average_reshaped_list.append(average_reshaped)
        average_x_list.append(average_x)
        average_y_list.append(average_y)
        n_replicates.append(len(image_names))

    dictionary = {"type": groups, "average_I": averages_list,
                  "average_I_reshaped": average_reshaped_list, "x": average_x_list,
                  "y": average_y_list, "n_replicates": n_replicates}

    df_averages = pd.DataFrame(dictionary)

    return df_averages


def PCA_analysis_averages(df_images, df_averages, n_components,  mask_filename):
    """
    This function reduces the dimensionality of a dataset of images using a principal
    component analysis. It uses an additional dataframe containing the reshaped
    averages of groups of images to define the pca components. Then, it opens
    each image in the images dataframe, project them in the PCA space and keeps
    the first n_components.

    Parameters
    ----------
    df_images : dataframe
    df_averages: dataframe
    n_components : int
        number of pca components to keep.
    mask_filename: str
        full path to a binary mask image.
    Returns
    -------
    data : dataframe
        updated dataframe with a column PCA_coefficients.
    data_eigenvectors : dataframe
        new dataframe containing the definition of each eigenvector and the
        portion of variance explained by each component.

    """

    print("Start PCA analysis")

    mask = np.asarray(Image.open(mask_filename))
    mask = mask/np.max(mask)

    averages_reshaped_list = df_averages["average_I_reshaped"]

    all_averages_stack = np.vstack(averages_reshaped_list)

    eigenvector0 = np.transpose(
        np.mean(np.stack(all_averages_stack, axis=1), axis=1))

    pca = PCA(n_components=n_components)
    pca.fit(all_averages_stack)

    pca_components = pca.components_
    explained_variance = pca.explained_variance_ratio_

    df_images.reset_index(inplace=True)

    df_images['PCA_coefficients'] = None

    for idx, row in df_images.iterrows():
        tmp_img, _, _ = aux_open_and_reshape_for_PCA(row["file name"], mask)
        [tmp_img_pca] = pca.transform([tmp_img])
        df_images.at[idx, "PCA_coefficients"] = tmp_img_pca

    data_eigenvectors = {"Eigenvector_0": eigenvector0, "ExplVar_0": 0}

    for i, _ in enumerate(pca_components):

        data_eigenvectors["Eigenvector_"+str(i+1)] = pca_components[i]
        data_eigenvectors["ExplVar_"+str(i+1)] = explained_variance[i]

    data_eigenvectors["x"] = df_averages.at[0, "x"]
    data_eigenvectors["y"] = df_averages.at[0, "y"]

    return df_images, data_eigenvectors


def PCA_analysis(data, n_components, column_for_PCA='I_reshaped'):
    """
    This funciton reduces the dimensionality of a dataset using a principal
    component analysis.

    Parameters
    ----------
    data : dataframe
    n_components : int
        number of pca components to keep.
    column_for_PCA : str, optional
        name of the dataframe column containing the data to analyze in the PCA.
        The default is 'I_reshaped'.

    Returns
    -------
    data : dataframe
        updated dataframe with a column PCA_coefficients.
    data_eigenvectors : dataframe
        new dataframe containing the definition of each eigenvector and the
        portion of variance explained by each component.

    """

    print("Start PCA analysis")

    all_images_list = data[column_for_PCA].values.tolist()

    all_images = np.vstack(all_images_list)

    eigenvector0 = np.transpose(np.mean(np.stack(all_images, axis=1), axis=1))

    pca = PCA(n_components=n_components)

    pca.fit(all_images)

    images_in_PCA_space = pca.transform(all_images)

    pca_components = pca.components_
    explained_variance = pca.explained_variance_ratio_

    data['PCA_coefficients'] = list(images_in_PCA_space)

    data_eigenvectors = {"Eigenvector_0": eigenvector0, "ExplVar_0": 0}

    for i, _ in enumerate(pca_components):

        data_eigenvectors["Eigenvector_"+str(i+1)] = pca_components[i]
        data_eigenvectors["ExplVar_"+str(i+1)] = explained_variance[i]

    data_eigenvectors["x"] = data.at[0, "x"]
    data_eigenvectors["y"] = data.at[0, "y"]

    return data, data_eigenvectors


def plot_PCA_2D(df_images, data_eigenvectors, comp1, comp2, hue='type',
                figsize=(14, 20), reverse_x=False, reverse_y=False, cmap="jet"):
    """
    This function is used to plot the representation of each image in a bidimensional
    space defined by two selected pca components. It also plots the image space
    representation of the two components (eigenvectors) on the two axes.

    Parameters
    ----------
    df_images : dataframe
    data_eigenvectors : dictionary
    comp1 : int
    comp2 : int
    hue : str, optional
        which column to use as hue in the plot. The default is 'type'.
    figsize : tuple, optional
        size of the figure. The default is (14, 20).
    reverse_x : bool, optional
        whether to invert the orientation of the x axis. The default is False.
    reverse_y : bool, optional
        whether to invert the orientation of the y axis. The default is False.

    Returns
    -------
    None.

    """
    # Create a grid to accomodate the scatterplot of PCA coefficients and
    # the plots of the relative eigenvectors in the image space on the axes.

    df_PCA = df_images[["PCA_coefficients", hue]].copy() 

    fig = plt.figure(constrained_layout=True, figsize=figsize)
    widths = [1, 4, 1]
    heights = [1, 4, 1]
    gs = fig.add_gridspec(
        ncols=3, nrows=3, width_ratios=widths, height_ratios=heights)

    f_ax1 = fig.add_subplot(gs[:-1, 1:])
    f_ax2 = fig.add_subplot(gs[-1:, -1:])
    f_ax3 = fig.add_subplot(gs[0, 0])
    f_ax4 = fig.add_subplot(gs[-1, 0])

    df_PCA['PCA_a'] = None
    df_PCA['PCA_b'] = None

    for i, row in df_images.iterrows():
        x = row['PCA_coefficients'][comp1]
        y = row['PCA_coefficients'][comp2]
        if reverse_x:
            x = -x
        if reverse_y:
            y = -y
        df_PCA.loc[i, 'PCA_a'] = x
        df_PCA.loc[i, 'PCA_b'] = y

    eigen0 = coo_matrix((data_eigenvectors["Eigenvector_0"], (
        data_eigenvectors["x"], data_eigenvectors["y"]))).toarray()
    eigen1 = coo_matrix((data_eigenvectors["Eigenvector_"+str(comp1)], (
        data_eigenvectors["x"], data_eigenvectors["y"]))).toarray()
    eigen2 = coo_matrix((data_eigenvectors["Eigenvector_"+str(comp2)], (
        data_eigenvectors["x"], data_eigenvectors["y"]))).toarray()

    expl_var1 = data_eigenvectors["ExplVar_"+str(comp1)]
    expl_var2 = data_eigenvectors["ExplVar_"+str(comp2)]

    sns.scatterplot(data=df_PCA, x='PCA_a', y='PCA_b',  hue=hue, ax=f_ax1)

    if reverse_x:
        eigen1 = -eigen1
    if reverse_y:
        eigen2 = -eigen2

    eigen0[eigen0 == 0] = np.nan
    eigen1[eigen1 == 0] = np.nan
    eigen2[eigen2 == 0] = np.nan

    vmax = max([np.nanmax(abs(eigen1)), np.nanmax(abs(eigen2))])

    f_ax2.imshow(eigen1, cmap=cmap)
    f_ax2.get_images()[0].set_clim(-vmax, vmax)
    f_ax2.axis('off')

    f_ax3.imshow(eigen2, cmap=cmap)
    f_ax3.get_images()[0].set_clim(-vmax, vmax)
    f_ax3.axis('off')

    f_ax4.imshow(eigen0, cmap=cmap)

    vmax = np.nanmax(abs(eigen0))

    f_ax4.get_images()[0].set_clim(-vmax, vmax)
    f_ax4.axis('off')
    f_ax4.set_title("average:")

    sns.set_context("talk")
    sns.set_style("whitegrid")
    sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
    sns.despine(offset=15)
    f_ax1.set(xlabel='PCA Component %d (%.2f)' % (comp1, expl_var1),
              ylabel='PCA Component %d (%.2f)' % (comp2, expl_var2))
    f_ax1.legend(bbox_to_anchor=(1.05, 1.0),
                 loc='upper left', ncol=1, fontsize=14)
    plt.show()
    return


def change_basis_and_plot_PCA_2D(df_images, data_eigenvectors, comp1, comp2, hue='type',
                       v1_x=1, v1_y=0, v2_x=0, v2_y=1, origin="empty", figsize=(14, 20), cmap="jet"):
    """
    his function is used to plot the representation of each image in a bidimensional
    space defined by a linear combination of two selected pca components.
    It also plots the image space representation of the two components
    (combinations of the two eigenvectors) next to the scatterplot axes.
    The new directions in the PCA space are defined by two unit vectors v1 and v2.

    Parameters
    ----------
    df_images : dataframe
    data_eigenvectors : dictionary
    comp1 : int
    comp2 : int
    hue : str, optional
        which column to use as hue in the plot. The default is 'type'.
    v1_x : float, optional
        The default is 1.
    v1_y : float, optional
        The default is 0.
    v2_x : float, optional
        The default is 0.
    v2_y : float, optional
         The default is 1.
    origin : str, optional
        which group of images to use as a new origin of the plot. The default is "empty".
    figsize : tuple, optional
        size of the figure. The default is (14, 20).
    cmap : TYPE, optional
        DESCRIPTION. The default is "jet".

    Returns
    -------
    None.

    """
    
    df_PCA = df_images[["PCA_coefficients", hue]].copy() 

    # Create the matrix for the change of basis:
    M = np.array([[v1_x, v2_x], [v1_y, v2_y]])
    M_inv = np.linalg.inv(M)

    # Create a grid to accomodate the scatterplot of PCA coefficients and
    # the plots of the relative eigenvectors in the image space on the axes.

    fig = plt.figure(constrained_layout=True, figsize=figsize)
    widths = [1, 4, 1]
    heights = [1, 4, 1]
    gs = fig.add_gridspec(
        ncols=3, nrows=3, width_ratios=widths, height_ratios=heights)

    f_ax1 = fig.add_subplot(gs[:-1, 1:])
    f_ax2 = fig.add_subplot(gs[-1:, -1:])
    f_ax3 = fig.add_subplot(gs[0, 0])
    f_ax4 = fig.add_subplot(gs[-1, 0])

    # Calculate new origin:
    x_origin = np.mean(np.stack(
        df_PCA.loc[df_PCA[hue] == origin]['PCA_coefficients'].values)[:, comp1])
    y_origin = np.mean(np.stack(
        df_PCA.loc[df_PCA[hue] == origin]['PCA_coefficients'].values)[:, comp2])

    for i, row in df_PCA.iterrows():
        x = row['PCA_coefficients'][comp1] - x_origin
        y = row['PCA_coefficients'][comp2] - y_origin

        [x, y] = np.matmul(M_inv, np.stack((x, y)))

        df_PCA.loc[i, 'PCA_a'] = x
        df_PCA.loc[i, 'PCA_b'] = y

    eigen1 = coo_matrix((data_eigenvectors["Eigenvector_"+str(comp1)], (
        data_eigenvectors["x"], data_eigenvectors["y"]))).toarray()
    eigen2 = coo_matrix((data_eigenvectors["Eigenvector_"+str(comp2)], (
        data_eigenvectors["x"], data_eigenvectors["y"]))).toarray()

    expl_var1 = data_eigenvectors["ExplVar_"+str(comp1)]
    expl_var2 = data_eigenvectors["ExplVar_"+str(comp2)]

    sns.scatterplot(data=df_PCA, x='PCA_a', y='PCA_b',  hue=hue, ax=f_ax1)

    eigen1_new = v1_x*eigen1+v1_y*eigen2
    eigen2_new = v2_x*eigen1+v2_y*eigen2

    # Set pixels==0 to nan so that they will be transparent:

    eigen1_new[eigen1_new == 0] = np.nan
    eigen2_new[eigen2_new == 0] = np.nan

    vmax = max([np.nanmax(abs(eigen1_new)), np.nanmax(abs(eigen2_new))])

    f_ax2.imshow(eigen1_new, cmap=cmap)
    f_ax2.get_images()[0].set_clim(-vmax, vmax)
    f_ax2.axis('off')

    f_ax3.imshow(eigen2_new, cmap=cmap)
    f_ax3.get_images()[0].set_clim(-vmax, vmax)
    f_ax3.axis('off')

    f_ax4.axis('off')
    f_ax4.set_title("")

    sns.set_context("talk")
    sns.set_style("whitegrid")
    sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
    sns.despine(offset=15)

    if (abs(v1_x) == 1) & (abs(v1_y) == 0) & (abs(v2_x) == 0) & (abs(v2_y) == 1):
        f_ax1.set(xlabel='PCA Component %d (%.2f)' % (comp1, expl_var1),
                  ylabel='PCA Component %d (%.2f)' % (comp2, expl_var2))
    else:
        f_ax1.set(xlabel='Mixed Component 1', ylabel='Mixed Component 2')

    f_ax1.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', ncol=2)
    plt.show()

    return


if __name__ == '__main__':

    preprocessed_folder = "../../data/07_masked_and_smooth"
    filename = 'DatasetInfo.csv'
    dataframe = pd.read_csv(os.path.join(preprocessed_folder, filename))
    mask_filename = '../../data/07_masked_and_smooth/mask_C2.tif'

    dataframe = dataframe[dataframe["channel"]=="C2"]
    dataframe_averages = reshape_averages_for_PCA(dataframe, mask_filename)
    dataframe, eigenvectors_dict = PCA_analysis_averages(
        dataframe, dataframe_averages, 4,  mask_filename)

    #dataframe = reshape_all_images_for_PCA(dataframe, mask_filename)
    #dataframe, eigenvectors_dict = PCA_analysis(dataframe, n_components=4)

    #plot_PCA_2D(dataframe, eigenvectors_dict, 1, 2, hue='type',
    #            figsize=(14, 20), reverse_x=False, reverse_y=False, cmap="jet")

    change_basis_and_plot_PCA_2D(dataframe, eigenvectors_dict, 1, 2, hue='type',
                       v1_x=1, v1_y=0, v2_x=0, v2_y=1, origin="empty", figsize=(14, 20), cmap="jet")
