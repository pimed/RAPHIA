"""
Author: Simon John Christoph Soerensen, MD (simonjcs@stanford.edu)
Created: February 16, 2021
Latest Version: June 16, 2021
Use this code to run the ProGNet prostate whole gland segmentation pipeline
Specify input and outputs in the bottom of the code
"""

from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import datetime
import numpy as np
import pandas as pd
import math
import os
import glob
import pydicom
import re
import shutil
import distutils.dir_util
import SimpleITK as sitk
import sys
from matplotlib import pyplot as plt
from itertools import chain
import tensorflow as tf
import matplotlib.pyplot as plt
import logging
import nibabel as nib
from glob import glob
from scipy.interpolate import interp1d
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from sklearn import metrics
from sklearn.metrics import auc as skauc
from scipy import interp
from scipy.ndimage import gaussian_filter
from time import strftime
import pickle
import vtk
import pandas
import collections
from pydicom_seg import MultiClassWriter
from pydicom_seg.template import from_dcmqi_metainfo
from pyntcloud import PyntCloud
from skimage.morphology import convex_hull_object
import trimesh, struct
from random import randint

import os
import sys
from argparse import ArgumentParser

### import self-defined functions
from ProGNet_model import *

tf.keras.backend.set_floatx('float32')


def build_parser():
    parser = ArgumentParser()
    
    # Paths
    parser.add_argument('--inputFile', type=str, default='/home/weishao/Desktop/pimed/results2/RadPathFusion/Wei_Deep_Learning_Registration/Pipeline/model/ProGNet/test/17437425_20200521_0_T2.mha', help='path to foler of training images')
    parser.add_argument('--outputDir', type=str, default='/home/weishao/Desktop/pimed/results2/RadPathFusion/Wei_Deep_Learning_Registration/Pipeline/model/ProGNet/test/', help='path to csv file of training examples')
    parser.add_argument('--standardHist', type=str, default='/home/weishao/Desktop/pimed/results2/RadPathFusion/Wei_Deep_Learning_Registration/Pipeline/model/ProGNet/std_hist_T2.npy', help='path to csv file of training examples')
    parser.add_argument('--modelPath', type=str, default='/home/weishao/Desktop/pimed/results2/RadPathFusion/Wei_Deep_Learning_Registration/Pipeline/model/ProGNet/prognet_t2.h5', help='path to trained models folder')
    parser.add_argument('--gpu-id', type=int, default=1, help='which gpu to use')
    return parser
   
   
def main():
    parser = build_parser()
    args = parser.parse_args()
    
    devices = tf.config.experimental.list_physical_devices('GPU')
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    tf.config.experimental.set_visible_devices(devices[args.gpu_id], 'GPU')
    
    ProGNet(args.inputFile, args.outputDir, args.standardHist, args.modelPath)
    

    
if __name__ == '__main__':
    main()
