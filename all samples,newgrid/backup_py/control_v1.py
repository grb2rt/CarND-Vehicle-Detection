
# My stuff:
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
import lesson_functions

color_space = 'HSV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 24    # Number of histogram bins
orient = 8  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
# Example: If all features are extracted with:
# (image, 'HSV', (16,16), 24, 8, 8, 2, 'ALL', True, True, True)
# the final size is 5544 features per image
''' debug:
image_test = '../trainset/non-vehicles/Extras/extra1.png'
#image_test = ('../trainset/non-vehicles/Extras/extra1.png', '../trainset/non-vehicles/Extras/extra2.png')
features = lesson_functions.extract_features(image_test, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)
#print(len(features))
#print(features[0].shape)
#print(features)
'''

# My stuff
# function returns all regions to search in a 1280x720px image
import makegrid
windows = makegrid.makegrid()
''' debug
# print(windows)
'''


#######################
# Udacity code
#######################


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

# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split


# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
#        print(window)
#        print(str(window[1][1])+':'+str(window[0][1])+' , '+str(window[0][0])+':'+str(window[1][0]))
        #3) Extract the test window from original image Format: ((x,y),(x,y)) my window = ((527, 440), (591, 376)), Udacity window = ((1104, 592), (1200, 688))
#        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        test_img = cv2.resize(img[window[1][1]:window[0][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = lesson_functions.extract_features2(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        # Add step to adjust format of lesson_functions.extract_features2
        features = np.concatenate(features)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows
    
    
# Read in cars and notcars
cars_list=[]
cars1 = glob.glob('../trainset/vehicles/GTI_Far/*.png')
cars2 = glob.glob('../trainset/vehicles/GTI_Right/*.png')
cars3 = glob.glob('../trainset/vehicles/GTI_MiddleClose/*.png')
cars4 = glob.glob('../trainset/vehicles/KITTI_extracted/*.png')
cars_list.extend(cars1)
cars_list.extend(cars2)
cars_list.extend(cars3)
cars_list.extend(cars4)
cars = cars_list

print('carslist: ')
print(len(cars_list))

nocars_list=[]
notcars1 = glob.glob('../trainset/non-vehicles/GTI/*.png')
nocars_list.extend(notcars1)
notcars2 = glob.glob('../trainset/non-vehicles/Extras/*.png')
nocars_list.extend(notcars2)
notcars = nocars_list

print('nocarslist: ')
print(len(nocars_list))

#for image in images:
#    if 'image' in image or 'extra' in image:
#        notcars.append(image)
#    else:
#        cars.append(image)

# Reduce the sample size because
# The quiz evaluator times out after 13s of CPU time
sample_size = 500
cars = cars[0:sample_size]
notcars = notcars[0:sample_size]

'''
### TODO: Tweak these parameters and see how the results change.
color_space = 'HSV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 24    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [400, 720] # Min and max in y to search in slide_window()
'''

car_features = lesson_functions.extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = lesson_functions.extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()

image = mpimg.imread('../test_images/test1.jpg')
image = image.astype(np.float32)/255   
draw_image = np.copy(image)

'''
image_test = '../trainset/non-vehicles/Extras/extra1.png'
image = mpimg.imread(image_test)
image = image.astype(np.float32)/255   
'''

print('read in image max')
print(str(np.amax(image)))

''' just to make sure the conversion is correct
image_test = '../trainset/non-vehicles/Extras/extra1.png'
image_tmp = mpimg.imread(image_test)
print('test image max')
print(np.amax(image_tmp))
'''

# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
# image = image.astype(np.float32)/255
#cv2.imwrite('draw_image.jpg', draw_image)

'''
windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                    xy_window=(96, 96), xy_overlap=(0.5, 0.5))
'''

hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)                       

window_img = lesson_functions.draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=2)                    

cv2.imwrite('temp.jpg', window_img)

# mpimg.imsave(window_img, "temp.jpg")
# bbox-example-image print









