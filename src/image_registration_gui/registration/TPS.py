#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================
# The folllowing code has been adapated from py-thin-plate-spline, 
# originally developed by Christoph Heindl.
#
# Licensed under MIT License
# ============================================================

if __name__ == '__main__':
    from _TPS_helpers import *
    from PIL import Image
    import numpy as np
    from matplotlib import pyplot as plt
else:
    from ._TPS_helpers import *
    
from scipy import ndimage
    
def TPSwarping(img, c_src, c_dst, dshape = (512,512)):
    """
    NAME    	Registration.TPSwarping

    ============================================================

    Warps the image using the tps module.

    Parameters
    ----------
    img : image input
    c_src : coordinates of the landmarks on the original image
    c_dst : coordinates of the landmarks on the reference image
    dshape : size of the output image, (512,512) by default

    Returns
    -------
    warped : Warped image

    """
    if len(img.shape) > 2:
        
        raise TypeError("The img has multiple channels, TPSWarping only accepts 2D grayscale images.")
    
    dshape = dshape or img.shape

    theta = tps_theta_from_points(c_src, c_dst, reduced=True)
    grid = tps_grid(theta, c_dst, dshape)
    mapx, mapy = tps_grid_to_remap(grid, img.shape)
    mapping = np.array([mapy, mapx])
    
    return ndimage.map_coordinates(img, mapping , order=3)

def show_warped(img, warped):
    """
    Used only for testing. Plots the original and warped version of the test image.

    """
    fig, axs = plt.subplots(1, 2, figsize=(16,8))
    axs[0].axis('off')
    axs[1].axis('off')
    axs[0].imshow(img[...,::-1], origin='upper')
    axs[0].scatter(c_src[:, 0]*img.shape[1], c_src[:, 1]*img.shape[0], marker='+', color='black')
    axs[1].imshow(warped[...,::-1], origin='upper')
    axs[1].scatter(c_dst[:, 0]*warped.shape[1], c_dst[:, 1]*warped.shape[0], marker='+', color='black')
    plt.show()


if __name__ == '__main__':
    from pathlib import Path
    import os
    parent = Path(__file__).parents[2]
    img_path = os.path.join(os.path.join(parent,"test_data"),"image.png")
    img = Image.open(img_path)
    img = np.asarray(img)[:,:,2]
    
    c_src = np.array([
        [0.0, 0.0],
        [1., 0],
        [1, 1],
        [0, 1],
        [0.3, 0.3],
        [0.7, 0.7],
    ])
    
    c_dst = np.array([
        [0., 0],
        [1., 0],    
        [1, 1],
        [0, 1],
        [0.4, 0.4],
        [0.6, 0.6],
    ])
        
    warped = TPSwarping(img, c_src, c_dst, dshape=img.shape)
    show_warped(img, warped)

