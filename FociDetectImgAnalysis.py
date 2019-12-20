# -*- coding: utf-8 -*-

# diverse helper functions regarding image analysis

"""
Created on Mon Feb 25 21:24:17 2019

@author: Tim Hohmann et al. - "Evaluation of machine learning models for 
automatic detection of DNA double strand breaks after irradiation using a gH2AX 
foci assay", PLOS One, 2020
"""
import os

# For image analysis:
from skimage.filters import threshold_otsu
from skimage.io import imsave
import skimage.filters as sk_filt
from skimage.morphology import watershed,reconstruction, disk, remove_small_objects, binary_erosion, binary_dilation
from skimage.exposure import equalize_adapthist as adapthisteq    
from skimage.filters import median
from skimage.measure import label, regionprops
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

# import numpy:
import numpy as np
# import 2d convulution:
from scipy import signal

###############################################################################
# gets nuclei from gray scale images:
def get_nuclei(image,file_name,med_filt_size = 5,hist_eq_kernel = 35,local_dist_size = 45,erode_size = 15,dilate_size = 25,min_area = 4000, watershedding = False):
    '''
    Function segmenting nuclei in fluorescent images using otsus threshold and watershed transformation.
    
    Input parameters:
    image                   = input image (array), grayscale
    med_filt_size           = size of median filtering before application of adaptive histogram equalization
    hist_eq_kernel          = kernel size for adaptive histogram equalization
    local_dist_size         = neighborhood for peak detection in distance transformed image
    erode_size/dilate_size  = size of structering element for erosion/dilation
    min_area                = minimal size of a nucleus
    Returns: 
    regionprops             = region properties of the labeled regions/nuclei
    '''    
    # set current path:
    main_file_path = os.getcwd()
    # filter image:
    image = adapthisteq(median(image,disk(med_filt_size)),hist_eq_kernel,0.06)
     
    # Global otsu:
    global_thresh = threshold_otsu(image)
    nuclei = image > global_thresh
    
    # fill holes and make it to boolean afterwards:
    seed = np.copy(nuclei)
    seed[0:-1, 0:-1] = nuclei.max()      
    nuclei = reconstruction(seed, nuclei, method='erosion') >0.5
        
    # Remove small objects in the nucelus image:
    nuclei = remove_small_objects(nuclei,min_area)
    nuclei = binary_erosion(nuclei,disk(erode_size))
    nuclei = binary_dilation(nuclei,disk(dilate_size))     
    nuclei = remove_small_objects(nuclei,3*min_area)
    
    if watershedding: 
        # apply watershed transformation to separate touching nuclei:
        # Generate the markers as local maxima of the distance to the background
        distance = ndi.distance_transform_edt(nuclei)
        local_max = peak_local_max(distance, indices=False, footprint=np.ones((local_dist_size, local_dist_size)),labels=nuclei)
        markers = ndi.label(local_max)[0]
        label_image = watershed(-distance, markers, mask=nuclei,watershed_line = True)
    else:
        label_image = label(nuclei)
        
    # get regionprops:
    props = regionprops(label_image)   
    os.chdir("Single Nuclei")  
    # remove small labeled regions and save image:
    for i in range(len(props)-1,-1,-1):               
        if props[i].area < min_area:
            del(props[i])    
     
    imsave(file_name+".png", 255*np.uint8(nuclei))    
    os.chdir(main_file_path)    
    return props

    

###############################################################################

def FilterFunc(image,filt_size = 5,frequencies = 0.15, scaling_range = [0,5]):
    '''
    Apply various filters to an image and store the results in a liste of shape
    [x-dimension of image * y-dimension of image, number_of_filters]
    number_of_filters depends on the number of different filter sizes used
    
    Input parameters:
    image                   = input image (array), grayscale
    frequencies             = list of frequencies used for gabor filtering
    scaling_range           = scale_range used for frangi filter
    scaling_range[0]        = lower bound
    scaling_range[1]        = upper_bound
    filt_size               = list of filter neighborhood sizes used for all 
                              other filters used
    
    Returns: 
    output_im               = data containing the filtered images in the form 
                              column vectors (via filtered_image = reshape(-1,1))
                              
    Order of output_im, columnwise:
        original image
        scharr filter
        gabor filter
        frangi filter
        autolevel filter
        autolevel_percentile filter
        equalize filter
        gradient filter
        fradient_percentile filter
        maximum filter
        mean filter
        mean_percentile filter
        mean_bilateral filter
        median filter
        minimum filter
        modal filter
        tophat filter
        entropy filter
        anisotropy filter
        
    Keep in mind that the number of rows per filter is dependent on the number 
    of filter sizes used per filter.
        
    '''    
    output_im = []    
    # original image:
    filt_im = image.astype(float)
    output_im.append(filt_im.reshape(-1,1))
    
    # scharr filter:
    filt_im = sk_filt.scharr(image)
    output_im.append(filt_im.reshape(-1,1))    
    for freq in frequencies:
        for angle in range(0,100,10):
            # print(angle)
            # gabor filter:
            # rotate filter ims in 10 degree steps up to 90 degree to get all necessary information
            # add up these images and use them as filtered image:
            temp_im, unused = sk_filt.gabor(image, theta = angle, frequency = freq, n_stds = 3)
            
            if angle == 0:
                filt_im = temp_im
            else:
                filt_im = filt_im + temp_im               
            
        # convert to float
        filt_im = filt_im.astype(float)
        output_im.append(filt_im.reshape(-1,1))
   
    for curr_range in range(scaling_range[0],scaling_range[1]):         
        # frangi filter:
        filt_im = sk_filt.frangi(image,scale_range = (0,5), beta2 = 0.02,scale_step = 0.3, beta1 = 5.1)
        output_im.append(filt_im.reshape(-1,1))        
        
    for i in range(len(filt_size)):
        
        # autolevel filter -> stretches histogram from black to white:
        filt_im = sk_filt.rank.autolevel(image,disk(filt_size[i]))
        filt_im = filt_im.astype(float)
        output_im.append(filt_im.reshape(-1,1))
        
        # equalize image using histogram:
        filt_im = sk_filt.rank.equalize(image,disk(filt_size[i]))
        filt_im = filt_im.astype(float)
        output_im.append(filt_im.reshape(-1,1))
          
        # return local gradient:
        filt_im = sk_filt.rank.gradient(image,disk(filt_size[i]))
        filt_im = filt_im.astype(float)
        output_im.append(filt_im.reshape(-1,1))
                
        # maximum filter
        filt_im = sk_filt.rank.maximum(image,disk(filt_size[i]))
        filt_im = filt_im.astype(float)
        output_im.append(filt_im.reshape(-1,1))
        
        # mean filter
        filt_im = sk_filt.rank.mean(image,disk(filt_size[i]))
        filt_im = filt_im.astype(float)
        output_im.append(filt_im.reshape(-1,1))
               
        # minimum filter
        filt_im = sk_filt.rank.minimum(image,disk(filt_size[i]))
        filt_im = filt_im.astype(float)
        output_im.append(filt_im.reshape(-1,1))
        
        # modal filter -> returns most frequent value
        filt_im = sk_filt.rank.modal(image,disk(filt_size[i]))
        filt_im = filt_im.astype(float)
        output_im.append(filt_im.reshape(-1,1))
    
        # tophat filter
        filt_im = sk_filt.rank.tophat(image,disk(filt_size[i]))
        filt_im = filt_im.astype(float)
        output_im.append(filt_im.reshape(-1,1))
        
        # entropy filter
        filt_im = sk_filt.rank.entropy(image,disk(filt_size[i]))
        output_im.append(filt_im.reshape(-1,1))        
        
        # make anisotropy filter.
        temp_im = image.astype(float)        
        # create meshgrid with gaussian function:
        x = np.linspace(0,filt_size[i]-1,filt_size[i])
        y = np.linspace(0,filt_size[i]-1,filt_size[i])
        xv, yv = np.meshgrid(x, y)        
        sigma = 3
        h = np.exp(-((xv-(filt_size[i]-1)/2)**2 + (yv-(filt_size[i]-1)/2)**2)/(2*sigma**2))
        h = h/sum(sum(h))    
        # Get the image gradient direction with the Scharr operator:
        sch_op_x = (1/32) * np.array([[-3,0,3],[-10,0,10],[-3,0,3]])
        sch_op_y = (1/32) * np.array([[-3,-10,-3],[0,0,0],[3,10,3]])        
        d_im_x   = signal.convolve2d(temp_im,sch_op_x, boundary='symm', mode='same')
        d_im_y   = signal.convolve2d(temp_im,sch_op_y, boundary='symm', mode='same')
                # Get the convoluted images:
        conv_im_xx = signal.convolve2d(d_im_x**2,h,boundary='symm', mode='same');
        conv_im_yy = signal.convolve2d(d_im_y**2,h,boundary='symm', mode='same');
        conv_im_xy = signal.convolve2d(d_im_x*d_im_y,h,boundary='symm', mode='same');
        coherence_im =   ((conv_im_xx - conv_im_yy)**2 + 4*conv_im_xy**2) / (conv_im_xx + conv_im_yy)**2
        temp_im = (conv_im_xx + conv_im_yy)**2 < 5*10**-1
        coherence_im[temp_im] = 0
        output_im.append(coherence_im.reshape(-1,1))
        
    output_im = np.concatenate(output_im,axis = 1)   
    return output_im
    
###############################################################################
def GoToFiles(file_path,save_name = False):
    '''
    Go to specified folder named "file_path", get name sof all files and create a
    sub-folder named "save_name" in "file_path" if it does not exist
    '''
    # go to foci folder:
    os.chdir(file_path)     
    # Get all files in nuclei directory:
    # files = os.listdir(file_path) 
    included_extensions = [".png",".tif",".jpg",".bmp"]
    files = [fn for fn in os.listdir(file_path)
              if any(fn.endswith(ext) for ext in included_extensions)]
    
    # check if save path exists if not create it:
    if save_name != False:
        save_path = file_path+"\\"+ str(save_name)
        if not(os.path.isdir(save_path)):
            os.makedirs(save_path)  
       
    return files


