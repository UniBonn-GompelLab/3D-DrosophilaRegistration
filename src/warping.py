#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 19:54:40 2022

@author: ceolin
"""
import io
import os
from PIL import Image
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from tqdm import tqdm

import skimage.transform as scikit_tr
from skimage.transform import resize



def register(source_image, source_image_dsred, source_landmarks, target_landmarks, target_shape):
    """
    Parameters
    ----------
    source_image : numpy array
        DESCRIPTION.
    source_landmarks : dictionary
        DESCRIPTION.
    target_landmarks : dictionary
        DESCRIPTION.

    Returns
    -------
    registered_image : TYPE
        DESCRIPTION.

    """
    
    common_landmarks = set(target_landmarks.keys()).intersection( set(source_landmarks.keys()) )
    
    src_x = []
    src_y = []
    tgt_x = []
    tgt_y = []

    for ldmk in common_landmarks:
        [x,y] = ast.literal_eval( target_landmarks[ldmk] )
        tgt_x.append(x) 
        tgt_y.append(y)
        
        [x, y] = ast.literal_eval( source_landmarks[ldmk] )
        src_x.append(x) 
        src_y.append(y)
            
    pts_src = np.vstack((src_x, src_y)).T
    pts_tgt = np.vstack((tgt_x, tgt_y)).T
    
    ## find homography transformation with scikit
    tform_prj = scikit_tr.estimate_transform('projective', pts_src, pts_tgt)
    source_image = scikit_tr.warp(source_image, inverse_map=tform_prj.inverse, output_shape=target_shape, preserve_range=True)
    source_image_dsred = scikit_tr.warp(source_image_dsred, inverse_map=tform_prj.inverse, output_shape=target_shape, preserve_range=True)
        
    # Piecewise affine transformation after homography:
    pts_src = tform_prj(pts_src)
    
    #Add four new points to define a bounding box:
    min_x = np.min(tgt_x)-50
    max_x = np.max(tgt_x)+50
    
    min_y = np.min(tgt_y)-50
    max_y = np.max(tgt_y)+50
    
    vertices = [[min_x, min_y],[min_x, max_y], [max_x, min_y], [max_x, max_y]]
    pts_tgt_2 = np.vstack((pts_tgt, vertices))
    pts_src_2 = np.vstack((pts_src, vertices))
    
    tform_piece = scikit_tr.estimate_transform('piecewise-affine', pts_src_2, pts_tgt_2)
    registered_image = scikit_tr.warp(source_image , inverse_map=tform_piece.inverse, output_shape=target_shape, preserve_range=True)
    registered_image_dsred = scikit_tr.warp(source_image_dsred, inverse_map=tform_piece.inverse, output_shape=target_shape, preserve_range=True)
    
    return registered_image, registered_image_dsred


def apply_registration(row, df_model, target_shape, destination_folder):
    landmarks = df_model["name"].values
    
    source_landmarks = {ldmk:row[ldmk] for ldmk in landmarks if (row[ldmk] == row[ldmk])}
    target_landmarks = {ldmk:df_model.loc[df_model["name"] == ldmk, "target"].values[0] for ldmk in landmarks}
    
    source_image = np.asarray(Image.open(row["full path"]))
    source_image_dsred = np.asarray(Image.open(row["full path"].replace('_C1', '_C2')))
    
    registered_image, registered_image_dsred = register(source_image, source_image_dsred, source_landmarks, target_landmarks, target_shape)
    registered_image  = Image.fromarray(registered_image.astype(np.uint16))
    registered_image_dsred  = Image.fromarray(registered_image_dsred.astype(np.uint16))
    registered_image.save(os.path.join(destination_folder, row["file name"].replace('Projected_C1', 'Warped_C1')))
    registered_image_dsred.save(os.path.join(destination_folder, row["file name"].replace('Projected_C1', 'Warped_C2')))
    return  pd.Series( row["file name"].replace('Projected_C1', 'Warped_C2') )

def warping(df_model, df_files, df_landmarks, df_info, model_image_name, read_folder, destination_folder):
    pd.options.mode.chained_assignment = None 
    df_info = df_info.rename(columns={"filename_gfp": "file name"})

    if os.path.exists(destination_folder):
        pass
    else:
        os.mkdir(destination_folder)
    
    landmarks = df_model["name"].values
    counts = np.zeros( df_landmarks["file name"].values.shape )
    
    for ldmk in landmarks:
        counts += (df_landmarks[ldmk] == df_landmarks[ldmk])
    
    df_landmarks["ldmk_counts"] = counts
    df_landmarks = df_landmarks[df_landmarks["ldmk_counts"]>3]
    
    image_paths = [os.path.join(read_folder, filename) for filename in df_landmarks["file name"]]
    df_landmarks["full path"] = image_paths
    df_landmarks['file name C2']  = ""
    
    target_image_shape = np.asarray(Image.open(model_image_name)).shape
    tqdm.pandas()
    df_landmarks['file name C2'] = df_landmarks.progress_apply(lambda row : apply_registration(row, df_model, target_image_shape, destination_folder), axis = 1)
    
    df_final = pd.merge(df_landmarks[['file name', 'file name C2','ldmk_counts']],df_files[['file name','image quality', 'annotated']],on='file name', how='left')
    df_final = pd.merge(df_final, df_info[['file name', 'construct']], on='file name', how='left')
    
    df_final = df_final.drop('file name', axis=1)
    df_final = df_final.rename(columns={'file name C2':'file name'})
    df_final.to_csv(os.path.join(destination_folder, "DatasetInformation.csv"))
    return



if __name__ == "__main__":
    
    df_model = pd.read_csv("/media/ceolin/Data/Lab Gompel/Projects/Fly_Abdomens/data_2/05_landmarks/model/model_dataframe.csv")
    
    df_landmarks = pd.read_csv("/media/ceolin/Data/Lab Gompel/Projects/Fly_Abdomens/data_2/05_landmarks/annotation_projects/annotation_batch01/landmarks_dataframe.csv")
    df_files = pd.read_csv("/media/ceolin/Data/Lab Gompel/Projects/Fly_Abdomens/data_2/05_landmarks/annotation_projects/annotation_batch01/images_dataframe.csv")
    df_info = pd.read_excel("/media/ceolin/Data/Lab Gompel/Projects/Fly_Abdomens/data_2/04_projected/DatasetInformation.xlsx")
    
    model_image_name = "/media/ceolin/Data/Lab Gompel/Projects/Fly_Abdomens/data_2/05_landmarks/model/reference_image.tif"
    destination_folder = "/media/ceolin/Data/Lab Gompel/Projects/Fly_Abdomens/data_2/06_warped"
    projected_images_folder = "/media/ceolin/Data/Lab Gompel/Projects/Fly_Abdomens/data_2/04_projected"
    
    warping(df_model, df_files, df_landmarks, df_info, model_image_name, projected_images_folder, destination_folder)

   