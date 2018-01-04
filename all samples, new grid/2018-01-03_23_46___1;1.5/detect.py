
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
from sklearn.model_selection import train_test_split

import lesson_functions
import makegrid
#import detect


# Make a list of cars and non-cars
def make_learnlist(sample_size):

    # Read in cars and notcars
    cars_list=[]
    cars0 = glob.glob('../docu/mycars/*.png')
    cars1 = glob.glob('../trainset/vehicles/GTI_Far/*.png')
    cars2 = glob.glob('../trainset/vehicles/GTI_Right/*.png')
    cars3 = glob.glob('../trainset/vehicles/GTI_MiddleClose/*.png')
    cars4 = glob.glob('../trainset/vehicles/KITTI_extracted/*.png')
    cars_list.extend(cars0)
    cars_list.extend(cars1)
    cars_list.extend(cars2)
    cars_list.extend(cars3)
#    cars_list.extend(cars4)
    cars = cars_list

#    cars0_tmp = mpimg.imread(cars0[1])
#    cars1_tmp = mpimg.imread(cars1[1])
	
#    print('jpg read max: '+str(np.amax(cars0_tmp)))
#    print('png read max: '+str(np.amax(cars1_tmp)))

#    cv2.imwrite('mycar.png', cars0_tmp)
#    cv2.imwrite('theothercar.png', cars1_tmp)
#    cv2.imwrite('mycar255.png', cars0_tmp*255)
#    cv2.imwrite('theothercar255.png', cars1_tmp*255)


    print('carslist: '+str(len(cars_list)))

    nocars_list=[]
    notcars0 = glob.glob('../docu/mynocars/*.png')
    nocars_list.extend(notcars0)
    notcars1 = glob.glob('../trainset/non-vehicles/GTI/*.png')
    nocars_list.extend(notcars1)
    notcars2 = glob.glob('../trainset/non-vehicles/Extras/*.png')
#    nocars_list.extend(notcars2)
    notcars = nocars_list

    print('nocarslist: '+str(len(nocars_list)))

    cars = cars[0:sample_size]
    notcars = notcars[0:sample_size]
	
    return cars, notcars


def train(car_features, notcar_features):

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
#    print('Using:',orient,'orientations',pix_per_cell,
#        'pixels per cell and', cell_per_block,'cells per block')
#    print('Feature vector length:', len(X_train[0]))
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
	
    return svc, X_scaler

# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
# image = image.astype(np.float32)/255
#cv2.imwrite('draw_image.jpg', draw_image)

def search_windows(img, windows, clf, scaler, color_space, 
                    spatial_size, hist_bins, orient, 
                    pix_per_cell, cell_per_block, 
                    hog_channel, spatial_feat, 
                    hist_feat, hog_feat):

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
        features = lesson_functions.extract_features2(test_img, 
                        color_space, 
                        spatial_size, 
                        hist_bins, 
                        orient, 
                        pix_per_cell, 
                        cell_per_block, 
                        hog_channel, 
                        spatial_feat, 
                        hist_feat, 
                        hog_feat)
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





