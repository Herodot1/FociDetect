# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 10:39:10 2019

@author: Tim Hohmann et al. - "Evaluation of machine learning models for 
automatic detection of DNA double strand breaks after irradiation using a gH2AX 
foci assay", PLOS One, 2020
"""


# main file to analyse new images using a pre trained model

###############################################################################
# Parameters and file path that have to be set manually:

# directory containing the foci images:
im_path_foci = "D:\\Sample Images\\foci"
# directory containing the dapi images:
im_path_dapi = "D:\\Sample Images\\dapi"
# path with the information for the trained model:
model_path = "D:\Python\FociDetection v2.0\Trained Models"
# file name of the model 
model_name = "MLP_Reduced_Trained"

# Parameters:
# min area of nucleus
min_area = 4000
# min foci area:
min_area_foci = 16
# color channel of nucleus. 0 = red, 1 = grenn, 2 = blue. for grayscale images
# this value is ignored.
nuc_chan = 2
# color channel of foc. 0 = red, 1 = grenn, 2 = blue. for grayscale images
# this value is ignored.
foci_chan = 1

# specify filter sizes. for best results these should be identical to the ones 
# used for model training.
# used filter sizes
filt_range = [2,10,30]
# scaling range for frangi filter
sc_range = list(range(2,11))
#frequency range for gabor filter
freq = [0.08,0.16,0.2]

# adjust image size (should be adjusted to match resolution of training images)
#image rescale factor:
rescale_factor =1.0

###############################################################################
###############################################################################
###############################################################################
# turn of warnings, this is especially annoying with sklearn stuff
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Get packages:
import pickle
# For directory, file handling etc.
import os
import sys
# for writing csv file:
import csv
# import pandas:
import pandas as pd
# import numpy
import numpy as np
# For image analysis:
from skimage.io import imsave, imread, imshow
from skimage.morphology import remove_small_objects
from skimage.transform import rescale
from skimage.color import rgb2gray
# add current path to the search path for python:
sys.path.append(os.getcwd())
main_file_path = os.getcwd()
# self written functions:
from FociDetectImgAnalysis import get_nuclei, GoToFiles
from GetFociData import get_labeled_data

# load model and scalings
files = GoToFiles(model_path)     
# load model: 
with open(model_name+".p", "rb") as fp:     #Unpickling
    model = pickle.load(fp)
# load scalings:
with open("STD_Scaler1.p", "rb") as fp:     #Unpickling
    s1 = pickle.load(fp)
with open("STD_Scaler2.p", "rb") as fp:     #Unpickling
    s2 = pickle.load(fp)
with open("PCA.p", "rb") as fp:             #Unpickling
    p = pickle.load(fp)
with open("MinMax_Scaler.p", "rb") as fp:   #Unpickling
    mnmx = pickle.load(fp)    
with open("idx.p", "rb") as fp:             #Unpickling
    idx = pickle.load(fp)    

###############################################################################
# Start analysis of dapi images
# assumes that the blue channel contains nuclei (line 94)
# go to nucleus folder:
print("Analyzing nucleus images ...")
os.chdir(im_path_dapi)
# start reading in image files:
stats = []        
# get file names
save_path = "Single Nuclei"        
files = GoToFiles(im_path_dapi,save_path)        
for file_num in range(len(files)):
    file_name, extension = os.path.splitext(files[file_num])
    # print(file_name + "   " + extension)
    if extension in [".png",".tif",".jpg",".bmp"]:         
        # read image:                 
        image = imread(files[file_num])
        image = rescale(image, rescale_factor, order=1,preserve_range = True)          
        image = np.uint8(image) 
        # get region props of the blue channel:
        if(len(image.shape)<3):
            stats.append(get_nuclei(image[:,:],file_name))
        else:
            stats.append(get_nuclei(image[:,:,nuc_chan],file_name))

# Get x and y data for model training and the coordinate for each image:
x_data, y_data, coords = get_labeled_data(im_path_foci,stats,filt_size = filt_range,freq = freq,scaling_range = sc_range, rescale_factor = rescale_factor,foci_chan = foci_chan)  
# here y-data is obsolete

###############################################################################
# analyze foci images:
print("Analyzing foci images with trained model ...")
foci_area = []
save_folder = "\\" + model_name + "_" + "Results"
files = GoToFiles(im_path_foci,save_folder)
for im in range(len(x_data)):
    print("Current Image:" + str(im+1))
    # create variables for test image:
    x_vals_im = pd.DataFrame(x_data[im])    
    # rescale data
    x_image_transformed = s1.transform(x_vals_im)
    # do PCA to reduce parameter number:
    x_image_transformed = p.transform(x_image_transformed)
    # cumulated sum of variance explained. take only data explaining 95% of
    # variance
    x_image_transformed = x_image_transformed[:,idx]
    x_image_transformed = s2.transform(x_image_transformed)
    x_image_transformed = mnmx.transform(x_image_transformed)
    #predict labels:
    y_pred = model.predict(x_image_transformed)     
    # read in foci image:
    image = imread(files[im])
    # check if rgb:
    if len(image.shape) >=3:
        image = rgb2gray(image)
    # rescale image:
    image = rescale(image, rescale_factor, order=1,preserve_range = True)
    # create boolean image:
    binary = np.full((image.shape[0], image.shape[1]), False, dtype=bool)    
    # temp_im = np.uint8(image.copy())  
    for i in range(len(y_pred)):    
        if y_pred[i] == True:
            binary[coords[im][i][0],coords[im][i][1]] = True              
    # remove small objects:
    binary = remove_small_objects(binary, min_size=min_area_foci)
    # temp_im = np.uint8(image.copy())  
    # temp_im[binary==1] = 255
    
    temp_im_rgb = np.full((image.shape[0], image.shape[1],3), False, dtype=bool) 
    temp_im_rgb = np.uint8(temp_im_rgb)
    # check if image is rgb:      
    temp_im_rgb[:,:,1] = np.uint8(image.copy())  
    temp_im_rgb[binary == 1,0] = 255
    temp_im_rgb[binary == 1,1] = 255
    temp_im_rgb[binary == 1,2] = 255
    # save resulting image:
    os.chdir(im_path_foci+ save_folder)
    save_name = "Im_"+ str(im) + "_" + model_name + ".png"     
    imsave(save_name,temp_im_rgb)  
    os.chdir(im_path_foci)   
    
    # calculate relative foci area in each nucleus:
    for obj in range(len(stats[im])):
        nuc_area = stats[im][obj].area
        coords_nuc = stats[im][obj].coords
        temp_var = binary[coords_nuc[:,0],coords_nuc[:,1]]
        unique, counts = np.unique(temp_var, return_counts=True)
        temp_var = dict(zip(unique, counts))        
        try:
            foci_area.append(temp_var[True]/nuc_area)
        except:
            foci_area.append(0)

# save relative foci area as .csv  
os.chdir(im_path_foci+ save_folder)    
save_name = model_name + '_Results.csv'
with open(save_name, mode='w') as res_file:    
    res_writer = csv.writer(res_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    res_writer.writerow(foci_area)
        
        
        
       
        
        
        
        