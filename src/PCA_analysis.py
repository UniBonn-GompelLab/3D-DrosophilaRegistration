##################################################
## Function(s) to do a PCA analysis on the spatial 
## distribution of different statistical properties
## of the single cell distribution of enhancer activity
## in Drosophila wings.
##################################################
## Author: Stefano
## Version: September/October 2021
##################################################


import pandas as pd
import numpy as np
from PIL import Image
from numpy import asarray
from matplotlib import pyplot as plt
from scipy.sparse import coo_matrix
from scipy import signal
from numpy.fft  import fft2, ifft2, ifftshift
from skimage.measure import block_reduce
from sklearn.decomposition import PCA
from skimage.measure import block_reduce
import seaborn as sns
import os


def reshape_for_PCA(data, mask_filename, normalize = False, saturation_threshold = None):
    
    # To avoid PCA capturing mainly the mask itself I have to reshape the images
    # and save them as lists of non-zero pixels for which I have to save x,y,intensity.
    # This is equivalent to save each image matrix in the sparse matrix coordinate
    # format.
    # The reshaped data are saved in new columns of the dataframe
    
    mask = np.asarray(Image.open(mask_filename))
    data['file name'] = data['folder']+data['file name']
    
    example_image = Image.open( data.loc[0,'file name'] )
    example_image  = np.asarray(example_image)
    
    binning = int(round(mask.shape[0]/ example_image.shape[0]))
    
    mask_binned = block_reduce(mask,block_size=(binning,binning),func=np.mean)
    mask_binned = (mask_binned>0.9).astype(np.uint8)
    
    print("Reshaping data for PCA")
    applied = data.apply(lambda row: aux_reshape_for_PCA(row['file name'], mask_binned, normalize=normalize, saturation_threshold = saturation_threshold), axis=1, result_type = 'expand')
    applied.columns = ['I_reshaped', 'x', 'y', 'saturated']
    data = pd.concat([data, applied], axis=1)
    if saturation_threshold:
        data = data[data['saturated']=='false']
    return data

def aux_reshape_for_PCA(image_file_name, mask, normalize, saturation_threshold):
    image = Image.open(image_file_name)
    image  = np.asarray(image)

    image = np.nan_to_num(image)
    image_min = np.min(image[np.nonzero(image)])
    image = (mask)*(image+image_min*(10**-6))
    sparse_image = coo_matrix(image)
    
    I = sparse_image.data
    saturated = 'undefined'
    
    if saturation_threshold:
        saturated_fraction = np.sum(I>saturation_threshold)/len(I)
        if saturated_fraction > 0.03:
            saturated = 'true'
        else:
            saturated = 'false'
    
    if normalize:
        I = I/np.mean(I)
        
    row = sparse_image.row
    col = sparse_image.col

    return I, row, col, saturated


def PCA_analysis(data, n_components, PCA_variable='I', option = "average"):
    
    # Function to perform PCA analysis/dimensionality reduction
    # the determination of PCA directions can be done on the entire dataset or 
    # on the dataset composed by the average expression of each construct.
    # The latter focuses on differences among constructs and does not assign 
    # weights to different constructs based on the numerosity of that dataset.
    
    print("Start PCA analysis")
    column = PCA_variable+"_reshaped"
    All_images_list = data[column].values.tolist()
    Number_of_replicates = np.cumsum([len(item) for item in All_images_list])[:-1]
    All_images = np.vstack(All_images_list)
    
    Eigenvector0 = np.transpose(np.mean(np.stack(All_images, axis=1), axis=1))
    
    pca = PCA(n_components=n_components)
    
    #if option == "all":
    pca.fit(All_images)
        
    #if option == "average":
    #    pca.fit(All_averages)

    # After PCA analysis is complete I get the projection of each image along 
    # each PCA direction, I also save the PCA components (eigenvectors) in the
    # dataframe for plotting.
    
    Images_In_PCA_space = pca.transform(All_images)
    PCA_components = pca.components_
    Explained_variance = pca.explained_variance_ratio_
    
    PCA_coefficients = list(Images_In_PCA_space)#np.split(Images_In_PCA_space, Number_of_replicates, axis=0)
    data[PCA_variable+'_PCA_coefficients'] = PCA_coefficients
    data[PCA_variable+'_PCA_eigenvectors'] = ""
    data[PCA_variable+'_PCA_explained_var'] = ""
    
    for i in data.index:
        data.at[i,PCA_variable+'_PCA_eigenvectors'] = np.vstack([Eigenvector0,PCA_components])
        data.at[i,PCA_variable+'_PCA_explained_var'] = Explained_variance
    
    return data

def plot_PCA_2D(data, comp1, comp2, PCA_variable='I', hue='construct', figsize=(14,20), reverse_x=False, reverse_y = False):
    # A long function to make a nice plot of two PCA components of the data
    # It also plots the image space representation of the two components on the
    # two axes.
    
    # Create a grid to accomodate the scatterplot of PCA coefficients and 
    # the plots of the relative eigenvectors in the image space on the axes.

    fig = plt.figure(constrained_layout=True, figsize=figsize)
    widths = [1, 4, 1]
    heights = [1, 4, 1]
    gs = fig.add_gridspec(ncols=3, nrows=3, width_ratios=widths, height_ratios=heights)
    
    f_ax1 = fig.add_subplot(gs[:-1,1:])
    f_ax2 = fig.add_subplot(gs[-1:, -1:])
    f_ax3 = fig.add_subplot(gs[0, 0])
    f_ax4 = fig.add_subplot(gs[-1, 0])
    
    Num_colors = len(data.index)
    
    # Reset the color cycler to zero and scatter plot, also prepare eigenvectors
    # for plotting
    #f_ax1.set_prop_cycle(custom_cycler)
    data['PCA_a'] = None
    data['PCA_b'] = None
    for i, row in data.iterrows():
        x = row[PCA_variable+'_PCA_coefficients'][comp1-1]
        y = row[PCA_variable+'_PCA_coefficients'][comp2-1]
        if reverse_x:
            x = -x
        if reverse_y:
            y = -y
        data.loc[i,'PCA_a'] = x
        data.loc[i,'PCA_b'] = y
        Eigen0 = coo_matrix((row[PCA_variable+'_PCA_eigenvectors'][0,:] , (row['x'], row['y']))).toarray()#[200:,100:]
        Eigen1 = coo_matrix((row[PCA_variable+'_PCA_eigenvectors'][comp1,:] , (row['x'], row['y']))).toarray()#[200:,100:]
        Eigen2 = coo_matrix((row[PCA_variable+'_PCA_eigenvectors'][comp2,:] , (row['x'], row['y']))).toarray()#[200:,100:]
        Expl_var1 = row[PCA_variable+'_PCA_explained_var'][comp1-1]
        Expl_var2 = row[PCA_variable+'_PCA_explained_var'][comp2-1]
    
    sns.scatterplot(data = data, x = 'PCA_a', y= 'PCA_b' ,  hue = hue, ax=f_ax1)
    
    if reverse_x:
            Eigen1 = -Eigen1
    if reverse_y:
            Eigen2 = -Eigen2
            
    Eigen0[Eigen0 == 0] = np.nan
    Eigen1[Eigen1 == 0] = np.nan
    Eigen2[Eigen2 == 0] = np.nan
    vmax = max([np.nanmax(abs(Eigen1)), np.nanmax(abs(Eigen2))])

    f_ax2.imshow(Eigen1, cmap = sns.color_palette("vlag", as_cmap=True))
    f_ax2.get_images()[0].set_clim(-vmax,vmax)
    f_ax2.axis('off')
    
    f_ax3.imshow(Eigen2, cmap = sns.color_palette("vlag", as_cmap=True))
    f_ax3.get_images()[0].set_clim(-vmax,vmax)
    f_ax3.axis('off')
    
    f_ax4.imshow(Eigen0, cmap = sns.color_palette("vlag", as_cmap=True))
    vmax = np.nanmax(abs(Eigen0))
    f_ax4.get_images()[0].set_clim(-vmax,vmax)
    f_ax4.axis('off')
    f_ax4.set_title("average:")
    #f_ax4.axis('off')
    #f_ax4.set_title("")
    
    
    sns.set_context("talk")
    sns.set_style("whitegrid")
    sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
    sns.despine(offset=15)
    f_ax1.set(xlabel='PCA Component %d (%.2f)' % (comp1, Expl_var1 ), ylabel='PCA Component %d (%.2f)' % (comp2, Expl_var2 ))
    f_ax1.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', ncol = 1, fontsize=14)
    plt.show()
    return



# def plot_PCA_2D(data, comp1, comp2, PCA_variable='I', figsize=(14,20)):
#     # A long function to make a nice plot of two PCA components of the data
#     # It also plots the image space representation of the two components on the
#     # two axes.
    
#     # Create a grid to accomodate the scatterplot of PCA coefficients and 
#     # the plots of the relative eigenvectors in the image space on the axes.

#     fig = plt.figure(constrained_layout=True, figsize=figsize)
#     widths = [1, 4, 1]
#     heights = [1, 4, 1]
#     gs = fig.add_gridspec(ncols=3, nrows=3, width_ratios=widths, height_ratios=heights)
    
#     f_ax1 = fig.add_subplot(gs[:-1,1:])
#     f_ax2 = fig.add_subplot(gs[-1:, -1:])
#     f_ax3 = fig.add_subplot(gs[0, 0])
#     f_ax4 = fig.add_subplot(gs[-1, 0])
    
#     Num_colors = len(data.index)
    
#     # Reset the color cycler to avoid same colors in the plot
#     from cycler import cycler
#     markers = ['o', '+', 'x', '*']
#     markers_cycle = [markers[i%len(markers)] for i in range(Num_colors)]
#     custom_cycler = (cycler(color=sns.color_palette("husl", Num_colors), marker = markers_cycle))
#     #f_ax1.set_prop_cycle(custom_cycler)
#     colors = custom_cycler.by_key()["color"]
#     markers = custom_cycler.by_key()["marker"]

#     # Plot lines connecting all data points of the same genotype to their average
#     for i, row, in data.iterrows():
#         x = row[PCA_variable+'_PCA_coefficients'][:,comp1-1]
#         y = row[PCA_variable+'_PCA_coefficients'][:,comp2-1]
#         x_cent = np.mean(x)
#         y_cent = np.mean(y)
#         for xx, yy in zip(x, y):
#             f_ax1.plot([x_cent, xx], [y_cent, yy], color = colors[i], marker = "")
            
#     # Reset the color cycler to zero and scatter plot, also prepare eigenvectors
#     # for plotting
#     #f_ax1.set_prop_cycle(custom_cycler)
#     for i, row in data.iterrows():
#         x = row[PCA_variable+'_PCA_coefficients'][:,comp1-1]
#         y = row[PCA_variable+'_PCA_coefficients'][:,comp2-1]       
#         sns.scatterplot(x = x, y=y , label = row["constructs"], ax=f_ax1, color = colors[i], marker = markers[i])
#         Eigen0 = coo_matrix((row[PCA_variable+'_PCA_eigenvectors'][0,:] , (row['x'], row['y']))).toarray()#[200:,100:]
#         Eigen1 = coo_matrix((row[PCA_variable+'_PCA_eigenvectors'][comp1,:] , (row['x'], row['y']))).toarray()#[200:,100:]
#         Eigen2 = coo_matrix((row[PCA_variable+'_PCA_eigenvectors'][comp2,:] , (row['x'], row['y']))).toarray()#[200:,100:]
#         Expl_var1 = row[PCA_variable+'_PCA_explained_var'][comp1-1]
#         Expl_var2 = row[PCA_variable+'_PCA_explained_var'][comp2-1]
    
    
    
#     Eigen0[Eigen0 == 0] = np.nan
#     Eigen1[Eigen1 == 0] = np.nan
#     Eigen2[Eigen2 == 0] = np.nan
#     vmax = max([np.nanmax(abs(Eigen1)), np.nanmax(abs(Eigen2))])

#     f_ax2.imshow(Eigen1, cmap = sns.color_palette("vlag", as_cmap=True))
#     f_ax2.get_images()[0].set_clim(-vmax,vmax)
#     f_ax2.axis('off')
    
#     f_ax3.imshow(Eigen2, cmap = sns.color_palette("vlag", as_cmap=True))
#     f_ax3.get_images()[0].set_clim(-vmax,vmax)
#     f_ax3.axis('off')
    
#     #f_ax4.imshow(Eigen0, cmap = sns.color_palette("vlag", as_cmap=True))
#     #vmax = np.nanmax(abs(Eigen0))
#     #f_ax4.get_images()[0].set_clim(-vmax,vmax)
#     #f_ax4.axis('off')
#     #f_ax4.set_title("average:")
#     f_ax4.axis('off')
#     f_ax4.set_title("")
    
    
#     sns.set_context("talk")
#     sns.set_style("whitegrid")
#     sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
#     sns.despine(offset=15)
#     f_ax1.set(xlabel='PCA Component %d (%.2f)' % (comp1, Expl_var1 ), ylabel='PCA Component %d (%.2f)' % (comp2, Expl_var2 ))
#     f_ax1.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', ncol = 3)
#     plt.show()
#     return

def plot_rotate_PCA_2D(data, comp1, comp2, PCA_variable='freq', hue='construct', v1_x=1, v1_y = 0, v2_x=0, v2_y=1, origin="empty", figsize=(14,20), cmap= "jet"):
    # A long function to make a nice plot of two PCA components of the data 
    # after a change of basis specified by two vectors and recentering the data 
    # on a specified phenotype.
    # It also plots the image space representation of the two components on the
    # two axes.

    # Create the change of basis matrix:
    M = np.array([[v1_x, v2_x],[v1_y, v2_y]])
    M_inv = np.linalg.inv(M)

    # Create a grid to accomodate the scatterplot of PCA coefficients and 
    # the plots of the relative eigenvectors in the image space on the axes.

    fig = plt.figure(constrained_layout=True, figsize=figsize)
    widths = [1, 4, 1]
    heights = [1, 4, 1]
    gs = fig.add_gridspec(ncols=3, nrows=3, width_ratios=widths, height_ratios=heights)

    f_ax1 = fig.add_subplot(gs[:-1,1:])
    f_ax2 = fig.add_subplot(gs[-1:, -1:])
    f_ax3 = fig.add_subplot(gs[0, 0])
    f_ax4 = fig.add_subplot(gs[-1, 0])

    Num_colors = len(data.index)

    # Reset the color cycler to avoid same colors in the plot
    from cycler import cycler
    markers = ['o', '+', 'x', '*']
    markers_cycle = [markers[i%len(markers)] for i in range(Num_colors)]
    custom_cycler = (cycler(color=sns.color_palette("husl", Num_colors), marker = markers_cycle))
    #f_ax1.set_prop_cycle(custom_cycler)
    colors = custom_cycler.by_key()["color"]
    markers = custom_cycler.by_key()["marker"]

    # Calculate new origin:
    x_origin = np.mean( np.stack(data.loc[data["construct"]==origin][PCA_variable+'_PCA_coefficients'].values)[:,comp1-1] )
    y_origin = np.mean( np.stack(data.loc[data["construct"]==origin][PCA_variable+'_PCA_coefficients'].values)[:,comp2-1] )
   
    # Reset the color cycler to zero and scatter plot, also prepare eigenvectors
    # for plotting
    #f_ax1.set_prop_cycle(custom_cycler)
    for i, row in data.iterrows():
        x = row[PCA_variable+'_PCA_coefficients'][comp1-1]-x_origin
        y = row[PCA_variable+'_PCA_coefficients'][comp2-1]-y_origin

        [x,y] = np.matmul(M_inv,np.stack((x,y)))
 
        data.loc[i,'PCA_a'] = x
        data.loc[i,'PCA_b'] = y
        Eigen0 = coo_matrix((row[PCA_variable+'_PCA_eigenvectors'][0,:] , (row['x'], row['y']))).toarray()#[200:,100:]
        Eigen1 = coo_matrix((row[PCA_variable+'_PCA_eigenvectors'][comp1,:] , (row['x'], row['y']))).toarray()#[200:,100:]
        Eigen2 = coo_matrix((row[PCA_variable+'_PCA_eigenvectors'][comp2,:] , (row['x'], row['y']))).toarray()#[200:,100:]
        Expl_var1 = row[PCA_variable+'_PCA_explained_var'][comp1-1]
        Expl_var2 = row[PCA_variable+'_PCA_explained_var'][comp2-1]

    sns.scatterplot(data = data, x = 'PCA_a', y= 'PCA_b' ,  hue = hue, ax=f_ax1)
    
    Eigen1_new = v1_x*Eigen1+v1_y*Eigen2
    Eigen2_new = v2_x*Eigen1+v2_y*Eigen2

    # Set pixels==0 to nan so that they will be transparent:

    Eigen1_new[Eigen1_new == 0] = np.nan
    Eigen2_new[Eigen2_new == 0] = np.nan 

    vmax = max([np.nanmax(abs(Eigen1_new)), np.nanmax(abs(Eigen2_new))])

    f_ax2.imshow(Eigen1_new, cmap = cmap)
    f_ax2.get_images()[0].set_clim(-vmax,vmax)
    f_ax2.axis('off')

    f_ax3.imshow(Eigen2_new, cmap = cmap)
    f_ax3.get_images()[0].set_clim(-vmax,vmax)
    f_ax3.axis('off')

    f_ax4.axis('off')
    f_ax4.set_title("")

    sns.set_context("talk")
    sns.set_style("whitegrid")
    sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
    sns.despine(offset=15)
    if (abs(v1_x)==1)&(abs(v1_y)==0)&(abs(v2_x)==0)&(abs(v2_y)==1):
        f_ax1.set(xlabel='PCA Component %d (%.2f)' % (comp1, Expl_var1 ), ylabel='PCA Component %d (%.2f)' % (comp2, Expl_var2 ))
    else:
        f_ax1.set(xlabel='Component 1', ylabel='Component 2' )
    
    f_ax1.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', ncol = 2)
    plt.show()

    return


if __name__ == '__main__':
    
    preprocessed_folder = "/media/ceolin/Data/Lab Gompel/Projects/Fly_Abdomens/data_2/07_masked_and_smooth"
    filename = 'DatasetInfo.csv'
    dataframe = pd.read_csv(os.path.join(preprocessed_folder,filename))
    mask_filename = '/media/ceolin/Data/Lab Gompel/Projects/Fly_Abdomens/data_2/07_masked_and_smooth/mask.tif'
    #dataframe = dataframe[dataframe["image quality"]=="good"]
    dataframe = reshape_for_PCA(dataframe, mask_filename)
        
    variable = "I"
    dataframe = PCA_analysis(dataframe, n_components=4, PCA_variable=variable, option="all")
    dataframe = dataframe.reset_index()
    plot_rotate_PCA_2D(dataframe, comp1=1, comp2=2, PCA_variable=variable, v1_x=-1, v1_y = 0, v2_x=0, v2_y=-1, origin="empty", figsize=(10,8))

