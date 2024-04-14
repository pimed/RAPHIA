# reference to https://github.com/CODAIT/deep-histopath/blob/master/deephistopath/wsi/filter.py

import glob
import numpy as np
from PIL import Image
import scipy.ndimage.morphology as sc_morph
import skimage.morphology as sk_morph
from scipy import ndimage as ndi
import argparse
import cv2

def blue_filter(np_img, red_thresh, green_thresh, blue_thresh, output_type="bool"):
    """
    np_img : color image as a numpy array
    output_type : type of array to return (bool, uint8, or image)
    """
    r = np_img[:,:,0] < red_thresh
    g = np_img[:,:,1] < green_thresh
    b = np_img[:,:,2] > blue_thresh   
    rg = (1.05 < np_img[:,:,1]/(np_img[:,:,0]+1e-5)) & (np_img[:,:,1]/(np_img[:,:,0]+1e-5) <1.20)
    rb = (1.10 < np_img[:,:,2]/(np_img[:,:,0]+1e-5)) & (np_img[:,:,2]/(np_img[:,:,0]+1e-5) <1.40)
    result = (r & g & b & rg & rb)
    
    if output_type=="bool":
        pass
    elif output_type=="uint8":
        result = result.astype("uint8")*255
    elif output_type=="image":
        result = Image.fromarray(result.astype("uint8")*255)
    else:
        raise ValueError("output_type not supported")
        
    return result

def black_filter(np_img, red_thresh, green_thresh, blue_thresh, output_type="bool"):
    """
    np_img : color image as a numpy array
    output_type : type of array to return (bool, uint8, or image)
    """
    r = np_img[:,:,0] < red_thresh
    g = np_img[:,:,1] < green_thresh
    b = np_img[:,:,2] < blue_thresh
    rg = (0.95 < np_img[:,:,1]/(np_img[:,:,0]+1e-5)) & (np_img[:,:,1]/(np_img[:,:,0]+1e-5) <1.05)
    rb = (0.95 < np_img[:,:,2]/(np_img[:,:,0]+1e-5)) & (np_img[:,:,2]/(np_img[:,:,0]+1e-5) <1.05)
    obj = (np_img[:,:,0] < 40) & (np_img[:,:,1] < 40) & (np_img[:,:,2] < 40)
    result = ((r & g & b & rg & rb) | obj)

    if output_type=="bool":
        pass
    elif output_type=="uint8":
        result = result.astype("uint8")*255
    elif output_type=="image":
        result = Image.fromarray(result.astype("uint8")*255)
    else:
        raise ValueError("output_type not supported")
    
    return result

def filter_binary_dilation(np_img, disk_size=5, iterations=1, output_type="uint8"):
    """
    np_img: binary image as a mumpy array
    output_type: type of array to return (bool, uint8, or image)
    """
    if np_img.dtype == "uint8":
        np_img = np_img / 255
        
    result = sc_morph.binary_dilation(np_img, sk_morph.disk(disk_size), iterations=iterations)
    
    if output_type == "bool":
        pass
    elif output_type=="uint8":
        result = result.astype("uint8")*255
    elif output_type=="image":
        result = Image.fromarray(result.astype("uint8")*255)
    else:
        raise ValueError("output_type not supported")
        
    return result

def filter_remove_small_objects(np_img, min_size=3000, output_type="uint8"):
    """
    np_img: numpy array of type bool
    min_size: minimum size of small object to remove
    output_type: type of array to return (bool, uint8, or image)
    """
    rem_sm = np_img.astype(bool)
    result = sk_morph.remove_small_objects(rem_sm, min_size=min_size)
    
    if output_type == "bool":
        pass
    elif output_type=="uint8":
        result = result.astype("uint8")*255
    elif output_type=="image":
        result = Image.fromarray(result.astype("uint8")*255)
    else:
        raise ValueError("output_type not supported")
        
    return result

def extract_ink(img):
    """
    directory: path to the folder containing .tiff images
    img_id: image index to analyse
    """
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    blue = blue_filter(img, 200, 200, 80, output_type="bool")
    blue = filter_binary_dilation(blue, disk_size=10, iterations=2, output_type="uint8")
    blue = filter_remove_small_objects(blue, min_size=10000, output_type="uint8")
    
    black = black_filter(img, 200, 200, 200, output_type="bool")
    black = filter_binary_dilation(black, disk_size=10, iterations=2, output_type="uint8")
    black = filter_remove_small_objects(black, min_size=20000, output_type="uint8")
    
    blue2blue = np.zeros([blue.shape[0], blue.shape[1], 3])
    blue2blue[:,:,2]=blue
    
    black2green = np.zeros([black.shape[0], black.shape[1], 3])
    black2green[:,:,1] = black
    
    ink = (blue2blue + black2green).astype("uint8")
    ink = cv2.cvtColor(ink, cv2.COLOR_BGR2RGB)
    
    return ink
        
