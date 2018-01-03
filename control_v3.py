
import matplotlib.image as mpimg
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from lesson_functions import *


import lesson_functions
# call lesson_functions.extract_features()
# Arguments
# 0. imgs --> A list of image path names to handle
# 1. color_space='RGB' ['RGB' , 'HSV' , 'LUV' , 'HLS' , 'YUV', 'YCrCb']
# 2. spatial_size=(32, 32) - for spatial binning
# 3. hist_bins=32 - number of historgram bins
# 4. orient=9 - for hog features
# 5. pix_per_cell=8 - for hog features
# 6. cell_per_block=2 - for hog features
# 7. hog_channel=0 [0, 1, 2, 'ALL']
# 8. spatial_feat=True [True, False] - turn on/off spatial binning
# 9. hist_feat=True [True, False] - turn on/off color histogram binning
# 10. hog_feat=True [True, False] - turn on/off HOG 

color_space = 'LUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 5    # Number of histogram bins
orient = 3  # HOG orientations
pix_per_cell = 16 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

# Example: If all features are extracted with:
# (image, 'HSV', (16,16), 24, 8, 8, 2, 'ALL', True, True, True)
# the final size is 5544 features per image

import makegrid
# function returns all regions to search in a 1280x720px image
windows = makegrid.makegrid()

import detect
# Creating lists and training model
# Call detect.make_learnlist with the parameter samle_size to
# get back a list of cars and non-cars
sample_size = 7880 # for making the carlist for training and validation

# For the output of a single image:
test_image = mpimg.imread('../test_images/test6.jpg')
test_image_name = 'test6_LUV.jpg'
test_image = test_image.astype(np.float32)/255   
draw_image = np.copy(test_image)



#######################
# Udacity code
#######################
    
# Create the training data
cars, notcars = detect.make_learnlist(sample_size)
# Extract the features
car_features = lesson_functions.extract_features(cars, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)
notcar_features = lesson_functions.extract_features(notcars, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)
# Learn  the features
svc, X_scaler = detect.train(car_features, notcar_features)
# Search for matches in image:
hot_windows = detect.search_windows(test_image, windows, svc, X_scaler, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)                       
# Drawing the identified rectangles on the image
window_img = lesson_functions.draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=2)                    
# document single image output
cv2.imwrite(test_image_name, window_img)








