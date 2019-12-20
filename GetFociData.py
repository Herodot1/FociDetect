# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 09:45:37 2019

@author: Tim Hohmann et al. - "Evaluation of machine learning models for 
automatic detection of DNA double strand breaks after irradiation using a gH2AX 
foci assay", PLOS One, 2020
"""

# diverse helper functions for obtaining the labeled data


# Get packages:
# For directory, file handling etc.
import os
# import numpy:
import numpy as np
# For image analysis:
from skimage.io import imread
from skimage.transform import rescale
# self written functions:
from FociDetectImgAnalysis import GoToFiles, FilterFunc

def get_labeled_data(im_path_foci,stats,im_path_foci_marked = False,filt_size = 5,freq = 0.15, scaling_range = [0,5],rescale_factor = 1,foci_chan = 1,mark_chan = 0):
    '''
    Function giving an array of values corresponding to the results of the 
    FociDetectImgAnalysis.FilterFunc function. This is given image wise and 
    only for pixels that are marked as foreground in
    FociDetectImgAnalysis.get_nuclei
    Currently assumes that the foci channel is the green channel in an rgb-
    image (line 80)
    
    Input parameters:
    im_path_foci            = path to folder containing images 
    im_path_foci_marked     = path to folder containing images that were marked 
                            manually in red channel. if not provided the marked 
                            region is set to the whole image segmented in 
                            FociDetectImgAnalysis.get_nuclei
    stats                   = region properties - output of 
                            FociDetectImgAnalysis.get_nuclei. size must match
                            the number of images in im_path_foci
                            
    Variables passed on to FociDetectImgAnalysis.FilterFunc:                         
    freq                    = list of frequencies used for gabor filtering
    scaling_range           = scale_range used for frangi filter
    scaling_range[0]        = lower bound
    scaling_range[1]        = upper_bound
    filt_size               = list of filter neighborhood sizes used for all 
                              other filters used                        

    Returns: 
    x_data                  = array of values corresponding to the results of 
                            the FociDetectImgAnalysis.FilterFunc function. 
    y_data                  = array of values corresponding to the manual 
                            labeling results (True, False only). if 
                            im_path_foci_marked is not provided it is equal to
                            stats.coords
    coords                  = coordinates of each point in the original image. 
                            shape corresponds to that of x_data and y_data
    '''
    
    # Get manually labeled images (marked with red) and extract labels:
    if im_path_foci_marked != False:
        marked_foci = get_foci_labels(im_path_foci_marked,rescale_factor,mark_chan)
    
    files = GoToFiles(im_path_foci)
    y_data = []
    x_data = []
    coords = []
    print("Generating feature space ...")
    # start reading in image files:      
    for file_num in range(len(files)):
        file_name, extension = os.path.splitext(files[file_num])
        print("Current Image:" + file_name + extension)
        if extension in [".png",".tif",".jpg",".bmp"]:                 
            # read image:                 
            image = imread(files[file_num])
            image = rescale(image, rescale_factor,preserve_range = True)
            image = np.uint8(image)
            # get foci channel:
            if(len(image.shape)==3):
                image = image[:,:,foci_chan] 
            
            temp_x_data = []
            temp_y_data = []
            temp_coords = []
            # create sub image containing nuclei only:
            if len(stats[file_num]) > 0:
                for obj in range(len(stats[file_num])):   
                    print("Current Image:" + file_name + extension + "  Current Nucleus:" + str(obj+1))    
                    # create sub images of nucles, foci and marked foci images:           
                    mask_nuc = stats[file_num][obj].image        
                    bounding = stats[file_num][obj].bbox
                    sub_image = image[bounding[0]:bounding[2],bounding[1]:bounding[3]].copy()      
                    # if no input path for im_path_foci_marked is given set it to sub_image
                    if im_path_foci_marked != False:
                        sub_image_marked = marked_foci[file_num][bounding[0]:bounding[2],bounding[1]:bounding[3]].copy()      
                    else:
                        sub_image_marked = sub_image.copy()
                                            
                    # implement it in such a way, that only in nuclear areas the whole data
                    # analysis is performed.
                    filtered_data = FilterFunc(sub_image,frequencies =  freq, scaling_range = scaling_range,filt_size = filt_size)
                                    
                    # create row vectors and filter everything that only pixels inside 
                    # each nucleus are used:
                    # reshape nucleus mask:
                    mask_nuc = mask_nuc.reshape(-1,1)
                    sub_image_marked = sub_image_marked.reshape(-1,1)
                    sub_image_marked = sub_image_marked[mask_nuc]
                    filtered_data = filtered_data[mask_nuc[:,0],:]              
                    # get image coordinates of each nucleus pixel:
                    coords_nuc = stats[file_num][obj].coords
                    
                    # to keep track of the files/objects: create data that is appended 
                    # inside one image (all objects of one image). Afterwards store it 
                    # to a variable outside of this loop.
                    # Do the same with the marked foci and coordinates for later usage.
                    if obj == 0:
                       temp_x_data = filtered_data.copy()
                       temp_y_data = sub_image_marked.copy()
                       temp_coords = coords_nuc.copy()
                    else:
                       temp_x_data = np.insert(temp_x_data,temp_x_data.shape[0],filtered_data,axis = 0) 
                       temp_y_data = np.insert(temp_y_data,temp_y_data.shape[0],sub_image_marked,axis = 0) 
                       temp_coords = np.insert(temp_coords,temp_coords.shape[0],coords_nuc,axis = 0) 
    
            x_data.append(temp_x_data.copy().tolist())
            y_data.append(temp_y_data.copy().tolist())
            coords.append(temp_coords.copy().tolist())
            
    return x_data, y_data, coords

###############################################################################
    
def get_foci_labels(im_path_foci_marked, rescale_factor = 1,mark_chan = 0):
    '''
    Function returning binary labels for image manually labeled with red.
    
    Input parameters:
    im_path_foci_marked     = path to folder containing images that were marked 
                            manually in red channel. if not provided the marked 
                            region is set to the whole image segmented in 
                            FociDetectImgAnalysis.get_nuclei
    
    Returns: 
    marked_foci             = boolean array containing True if pixel was marked 
                            in red and False otherwise 
    '''
    # Get labeld trainings files (for gods sake this has to be done only once):  
    # get file names
    files = GoToFiles(im_path_foci_marked)
    #foci_idx = []
    marked_foci = []
    # roi_foci = []
    # start reading in image files:      
    for file_num in range(len(files)):
        file_name, extension = os.path.splitext(files[file_num])
        #print(file_name + "   " + extension)
        if extension in [".png",".tif",".jpg",".bmp"]:                 
            # read image:                 
            marked_image = imread(files[file_num])
            marked_image = rescale(marked_image, rescale_factor,preserve_range = True)
            marked_image = np.uint8(marked_image)
            # get red channel:
            marked_image = marked_image[:,:,mark_chan]>200
            marked_foci.append(marked_image)
            
    return marked_foci