# -*- coding: utf-8 -*-
"""
    mfeat.io
    ~~~~~~~

    This module provides simple i/o routines for micrograph datasets

    :license: MIT, see LICENSE for more details.
"""

from skimage import io
import numpy as np

def load_image(image_path, barheight=0):
    # crop scale bars off Matt Hecht's SEM images
    # UHCS images have barheight=38 px
    image = io.imread(image_path)
    
    if np.max(image)<=1:
        image *= 255
        image = np.clip(image, 0, 255).astype('uint8')
    
    if barheight > 0:
        image = image[:-barheight,:]
    return image
