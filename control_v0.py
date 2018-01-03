

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

image_test = '../trainset/non-vehicles/Extras/extra1.png'
#image_test = ('../trainset/non-vehicles/Extras/extra1.png', '../trainset/non-vehicles/Extras/extra2.png')
features = lesson_functions.extract_features(image_test, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)

#print(len(features))
#print(features[0].shape)
#print(features)
