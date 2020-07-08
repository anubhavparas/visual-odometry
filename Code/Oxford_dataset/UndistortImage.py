from scipy.ndimage import map_coordinates as interp2
import numpy as np
import cv2

# UndistortImage - undistort an image using a lookup table
# 
# INPUTS:
#   image: distorted image to be rectified
#   LUT: lookup table mapping pixels in the undistorted image to pixels in the
#     distorted image, as returned from ReadCameraModel
#
# OUTPUTS:
#   undistorted: image after undistortion

################################################################################
#
# Copyright (c) 2019 University of Maryland
# Authors: 
#  Kanishka Ganguly (kganguly@cs.umd.edu)
#
# This work is licensed under the Creative Commons 
# Attribution-NonCommercial-ShareAlike 4.0 International License. 
# To view a copy of this license, visit 
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to 
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
################################################################################


def UndistortImage(image,LUT):
    reshaped_lut = LUT[:, 1::-1].T.reshape((2, image.shape[0], image.shape[1]))
    undistorted = np.rollaxis(np.array([interp2(image[:, :, channel], reshaped_lut, order=1)
                                for channel in range(0, image.shape[2])]), 0, 3)

    
    return undistorted.astype(image.dtype)