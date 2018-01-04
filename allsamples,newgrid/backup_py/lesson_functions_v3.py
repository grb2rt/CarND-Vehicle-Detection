
import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog

#import lesson_functions
import makegrid
import detect


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

		
# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

	
# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

	
# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
#def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
#                        hist_bins=32, orient=9, 
#                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
#                        spatial_feat=True, hist_feat=True, hog_feat=True):
def extract_features2(image, 
                        color_space, 
                        spatial_size, 
                        hist_bins, 
                        orient, 
                        pix_per_cell, 
                        cell_per_block, 
                        hog_channel, 
                        spatial_feat, 
                        hist_feat, 
                        hog_feat):
    # apply color conversion if other than 'RGB'
    file_features = []
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(image)      

    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        file_features.append(spatial_features)
    if hist_feat == True:
        # Apply color_hist()
        hist_features = color_hist(feature_image, nbins=hist_bins)
        file_features.append(hist_features)
    if hog_feat == True:
    # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        file_features.append(hog_features)
    # Return list of feature vectors
#    print('extract_features2 executed')
    return file_features

# build a function to detect single image imput and act accordingly
def extract_features(imgs, 
                        color_space, 
                        spatial_size, 
                        hist_bins, 
                        orient, 
                        pix_per_cell, 
                        cell_per_block, 
                        hog_channel, 
                        spatial_feat, 
                        hist_feat, 
                        hog_feat):
    # Create a list to append feature vectors to
    features = []
    imgs = np.asarray(imgs)
#    print(imgs.shape) 
    if imgs.shape==():
#        print('there it is!')
        image = mpimg.imread(str(imgs))    # RGB color by default !!!!        
        features = extract_features2(image, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)
        features = np.concatenate(features)
    else:
        # Iterate through the list of images
        for file in imgs:
            file_features = []
            # Read in each one by one
            image = mpimg.imread(file)    # RGB color by default !!!!
            file_features = extract_features2(image, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)
            features.append(np.concatenate(file_features))
    return features	



   

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 0.8), thick=2):
    # Make a copy of the image
    imcopy = np.copy(img)*255
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick, lineType=4 )
    # Return the image copy with boxes drawn
    return imcopy
#    return img