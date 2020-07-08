import numpy as np

def ReadCameraModel(models_dir):

# ReadCameraModel - load camera intrisics and undistortion LUT from disk
#
#
# INPUTS:
#   image_dir: directory containing images for which camera model is required
#   models_dir: directory containing camera models
#
# OUTPUTS:
#   fx: horizontal focal length in pixels
#   fy: vertical focal length in pixels
#   cx: horizontal principal point in pixels
#   cy: vertical principal point in pixels
#   G_camera_image: transform that maps from image coordinates to the base
#     frame of the camera. For monocular cameras, this is simply a rotation.
#     For stereo camera, this is a rotation and a translation to the left-most
#     lense.
#   LUT: undistortion lookup table. For an image of size w x h, LUT will be an
#     array of size [w x h, 2], with a (u,v) pair for each pixel. Maps pixels
#     in the undistorted image to pixels in the distorted image
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
    intrinsics_path = models_dir + "/stereo_narrow_left.txt"
    lut_path = models_dir + "/stereo_narrow_left_distortion_lut.bin"

    intrinsics = np.loadtxt(intrinsics_path)
    # Intrinsics
    fx = intrinsics[0,0]
    fy = intrinsics[0,1]
    cx = intrinsics[0,2]
    cy = intrinsics[0,3]
    # 4x4 matrix that transforms x-forward coordinate frame at camera origin and image frame for specific lens
    G_camera_image = intrinsics[1:5,0:4]
    # LUT for undistortion
    # LUT consists of (u,v) pair for each pixel)
    lut = np.fromfile(lut_path, np.double)
    lut = lut.reshape([2, lut.size//2])
    LUT = lut.transpose()

    return fx, fy, cx, cy, G_camera_image, LUT