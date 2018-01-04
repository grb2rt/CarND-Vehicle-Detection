
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from scipy.ndimage.measurements import label
import pickle


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
#spatial_size = (16, 16) # Spatial binning dimensions
spatial_size = (32, 32) # Spatial binning dimensions
#hist_bins = 13    # Number of histogram bins
hist_bins = 32    # Number of histogram bins
#orient = 15  # HOG orientations
orient = 7  # HOG orientations
pix_per_cell = 16 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

# Example: If all features are extracted with:
# (image, 'HSV', (16,16), 24, 8, 8, 2, 'ALL', True, True, True)
# the final size is 5544 features per image

dyn_lim = 0.1 # percentage of detections to locate object
abs_threshold = 7 # no of min detections to detect object
max_threshold = 12 # no of max detections to detect object
abs_limit = 0 # At least that amount +1 must be detected in any filter
 
#substart = 10.2; subend = 10.5 # single car, white, close
#substart = 26.5; subend = 26.8 # single car, white, far
#substart = 29.5; subend = 29.8 # 2 cars, white far, black close
#substart = 39.5; subend = 39.8 # 2 cars, white close, black far
#substart = 49.5; subend = 49.8 # single car, black, far
#substart = 33.5; subend = 33.8 # single car, black, far
#substart = 27; subend = 28 # critical situation
#substart = 23.6 ; subend = 24 # single white car, low abs_threshold needed due to extremely few numbers of windows
#substart = 50.2; subend = 50.5 # 2nd car approaching
#substart = 1 ; subend = 1.5 # idle street


substart = 0; subend = 50 # all


import makegrid
# function returns all regions to search in a 1280x720px image
windows = makegrid.makegrid()

import detect
# Creating lists and training model
# Call detect.make_learnlist with the parameter samle_size to
# get back a list of cars and non-cars
# sample_size = 7880 # for making the carlist for training and validation
sample_size = 9000 # for making the carlist for training and validation

import heatmap
# Creating the heatmap rectangles


train_model = 0  # If set, the model is trained again. Otherwise read in as data from previous run


#######################
# create documentation structure - goal: to have everything in one folder for documentation of the run
#######################

import datetime
import os

x = str(datetime.datetime.now())
folder = x[0:10]+"_"+x[11:13]+"_"+x[14:16] +'___'+str(substart)+';'+str(subend)
os.makedirs(folder)
os.makedirs(folder+"/spat/")
os.makedirs(folder+"/hist/")
os.makedirs(folder+"/hog/")
os.makedirs(folder+"/all/")
os.makedirs(folder+"/output/")
os.makedirs(folder+"/testpic/")

import shutil
shutil.copyfile('control.py', folder+'/control.py') 
shutil.copyfile('detect.py', folder+'/detect.py') 
shutil.copyfile('heatmap.py', folder+'/heatmap.py') 
shutil.copyfile('lesson_functions.py', folder+'/lesson_functions.py') 
shutil.copyfile('makegrid.py', folder+'/makegrid.py') 



#######################
# Train
#######################

if train_model == 1:
    # Create the training data
    cars, notcars = detect.make_learnlist(sample_size)
    # Extract the features
    car_features_spat = lesson_functions.extract_features(cars, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, False, False,)
    notcar_features_spat = lesson_functions.extract_features(notcars, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, False, False,)
    
    car_features_hist = lesson_functions.extract_features(cars, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, False)
    notcar_features_hist = lesson_functions.extract_features(notcars, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, False)
    
    car_features_hog  = lesson_functions.extract_features(cars, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, False, False, hog_feat)
    notcar_features_hog  = lesson_functions.extract_features(notcars, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, False, False, hog_feat)
    # Learn  the features
    svc_spat, X_scaler_spat = detect.train(car_features_spat, notcar_features_spat)
    svc_hist, X_scaler_hist = detect.train(car_features_hist, notcar_features_hist)
    svc_hog , X_scaler_hog  = detect.train(car_features_hog , notcar_features_hog )
    
    obj = svc_spat, X_scaler_spat, svc_hist, X_scaler_hist, svc_hog , X_scaler_hog
	
    # Saving the objects:
    f = open('trained_parameters.pkl', 'wb')
    pickle.dump(obj, f)
    f.close()
	#with open('trained_parameters.pkl', 'w') as f:  # Python 3: open(..., 'wb')
    #    pickle.dump([svc_spat, X_scaler_spat, svc_hist, X_scaler_hist, svc_hog , X_scaler_hog], f)
else:
    f = open('trained_parameters.pkl', 'rb')
    obj = pickle.load(f)
#    svc_spat; X_scaler_spat; svc_hist; X_scaler_hist; svc_hog ; X_scaler_hog = obj
    svc_spat = obj[0]
    X_scaler_spat = obj[1]
    svc_hist = obj[2]
    X_scaler_hist = obj[3]
    svc_hog = obj[4]
    X_scaler_hog = obj[5]
    f.close()
#    with open('trained_parameters.pkl') as f:  # Python 3: open(..., 'rb')
#    svc_spat, X_scaler_spat, svc_hist, X_scaler_hist, svc_hog , X_scaler_hog = pickle.load(f)
 
shutil.copyfile('trained_parameters.pkl', folder+'/trained_parameters.pkl') 


#######################
# Image flow
#######################


# For the output of a single image:
test_file_name = 'test6.jpg'
test_file_folder = '../test_images/'
test_file = test_file_folder+test_file_name
test_image_name = test_file_name+'_LUVonlySpat.jpg'
test_image_png = mpimg.imread(test_file)
test_image = test_image_png.astype(np.float32)/255
draw_image = np.copy(test_image)


# Search for matches in image:
hot_windows_spat = detect.search_windows(test_image, windows, svc_spat, X_scaler_spat, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=False    , hog_feat=False   )                       
hot_windows_hist = detect.search_windows(test_image, windows, svc_hist, X_scaler_hist, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=False   )                       
hot_windows_hog  = detect.search_windows(test_image, windows, svc_hog , X_scaler_hog , color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_feat=False       , hist_feat=False    , hog_feat=hog_feat)                       
#print("hot_windows_spat: "+str(len(hot_windows_spat)))
#print("hot_windows_hist: "+str(len(hot_windows_hist)))
#print("hot_windows_hog: " +str(len(hot_windows_hog )))
hot_windows = []
hot_windows.extend(hot_windows_spat)
hot_windows.extend(hot_windows_hist)
hot_windows.extend(hot_windows_hog )
burning_windows = len(hot_windows)


print("hot_windows found: " +str(len(hot_windows )))

# Drawing the identified rectangles on the image
window_img = lesson_functions.draw_boxes(draw_image, hot_windows_spat, color=(255, 0, 255), thick=2)                    
cv2.imwrite(folder+'/testpic/'+test_file_name+'_spat.jpg', window_img)
window_img = lesson_functions.draw_boxes(draw_image, hot_windows_hist, color=(0, 255, 255), thick=2)                    
cv2.imwrite(folder+'/testpic/'+test_file_name+'_hist.jpg', window_img)
window_img = lesson_functions.draw_boxes(draw_image, hot_windows_hog , color=(255, 255, 0), thick=2)                    
cv2.imwrite(folder+'/testpic/'+test_file_name+'_hog.jpg', window_img)
window_img = lesson_functions.draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=2)                    
cv2.imwrite(folder+'/testpic/'+test_file_name+'_All_criteria.jpg', window_img)
# document single image output

# consolidate heatmap
#threshold = 6
#threshold = 150
dyn_threshold = int(burning_windows*dyn_lim) # 15% of all frames must be overlapping object for detection
labels, heat_max = heatmap.heathot(hot_windows, test_image, dyn_threshold, abs_threshold, max_threshold)

draw_img = heatmap.draw_labeled_bboxes(test_image_png, labels)
cv2.imwrite(folder+'/testpic/'+test_file_name+'_heatmap.jpg', draw_img)


from moviepy.editor import VideoFileClip
from IPython.display import HTML
from PIL import Image

nnn = 0

def videopipe(video_image):
    # this is the way to treat each videoframe
    video_image_png = video_image.astype(np.float32)/255
    draw_image = np.copy(video_image)
#    print(str(np.amax(draw_image)))

    '''
    # For the output of a single image:
    test_file_name = 'test1.jpg'
    test_file_folder = '../test_images/'
    test_file = test_file_folder+test_file_name
    test_image_name = test_file_name+'_LUVonlySpat.jpg'
    test_image_png = mpimg.imread(test_file)
    test_image = test_image_png.astype(np.float32)/255
    draw_image = np.copy(test_image)
   ''' 
    
    # Search for matches in image:
    hot_windows_spat = detect.search_windows(video_image_png, windows, svc_spat, X_scaler_spat, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=False    , hog_feat=False   )                       
    hot_windows_hist = detect.search_windows(video_image_png, windows, svc_hist, X_scaler_hist, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=False   )                       
    hot_windows_hog  = detect.search_windows(video_image_png, windows, svc_hog , X_scaler_hog , color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_feat=False       , hist_feat=False    , hog_feat=hog_feat)                       
    #print("hot_windows_spat: "+str(len(hot_windows_spat)))
    #print("hot_windows_hist: "+str(len(hot_windows_hist)))
    #print("hot_windows_hog:  "+str(len(hot_windows_hog )))
    hot_windows = []
    hot_windows.extend(hot_windows_spat)
    hot_windows.extend(hot_windows_hist)
    hot_windows.extend(hot_windows_hog )
    burning_windows = len(hot_windows)

    font = cv2.FONT_HERSHEY_SIMPLEX

#    '''	# for debugging:
    # Drawing the identified rectangles on the image
    window_img = lesson_functions.draw_boxes(draw_image, hot_windows_spat, color=(255, 0, 255), thick=2)                    
    cv2.putText(window_img, str(nnn) , (50, 100), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(window_img, "hot windows found: "+str(len(hot_windows_spat)) , (150, 100), font, 1.5, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(window_img, "spatial bins: "+str(spatial_size) , (100, 150), font, 1, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite(folder+'/spat/spat_'+str(nnn)+'.jpg', window_img)

    window_img = lesson_functions.draw_boxes(draw_image, hot_windows_hist, color=(0, 255, 255), thick=2)                    
    cv2.putText(window_img, str(nnn) , (50, 100), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(window_img, "hot windows found: "+str(len(hot_windows_hist)) , (150, 100), font, 1.5, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(window_img, "hist_bins: "+str(hist_bins) , (100, 150), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(folder+'/hist/hist_'+str(nnn)+'.jpg', window_img)

    window_img = lesson_functions.draw_boxes(draw_image, hot_windows_hog , color=(255, 255, 0), thick=2)                    
    cv2.putText(window_img, str(nnn) , (50, 100), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(window_img, "hot windows found: "+str(len(hot_windows_hog)) , (150, 100), font, 1.5, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(window_img, "orient: "+str(orient)+"  pix_per_cell: "+str(pix_per_cell)+"  cell_per_block: "+str(cell_per_block)+"  hog_channel: "+str(hog_channel) , (100, 150), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(folder+'/hog/hog_'+str(nnn)+'.jpg', window_img)

    window_img = lesson_functions.draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=2)                    
    cv2.putText(window_img, str(nnn) , (50, 100), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(window_img, "hot windows found: "+str(len(hot_windows)) , (150, 100), font, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(window_img, "abs_threshold: "+str(abs_threshold) + "  dyn_lim: "+str(dyn_lim)+ " = "+str(int(dyn_lim*len(hot_windows))) , (100, 150), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(window_img, "spat: "+str(len(hot_windows_spat))+"  hist: "+str(len(hot_windows_hist))+"  hog: "+str(len(hot_windows_hog)) , (100, 200), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    # document single image output
#    '''

    global nnn
    nnn = nnn + 1
	
    # consolidate heatmap
#    threshold = 6
#    dyn_threshold = int(burning_windows*0.15) # % of all frames must be overlapping object for detection
    dyn_threshold = int(burning_windows*dyn_lim) # % of all frames must be overlapping object for detection
    labels, heat_max = heatmap.heathot(hot_windows, draw_image, dyn_threshold, abs_threshold, max_threshold)
#    print("labels shape after: "+labels.shape)
    if (len(hot_windows_spat) > abs_limit) and (len(hot_windows_hist) > abs_limit) and (len(hot_windows_hog) > abs_limit): # there have to be at least some detections in each filter
        draw_image = heatmap.draw_labeled_bboxes(draw_image, labels)
#    else:
#        draw_img = draw_image

#    ''' for documentation purposes add results to output image
    if (len(hot_windows_spat) > abs_limit) and (len(hot_windows_hist) > abs_limit) and (len(hot_windows_hog) > abs_limit): # there have to be at least some detections in each filter
        docu_img = heatmap.draw_labeled_bboxes(window_img, labels)
    else:
        docu_img = window_img
    cv2.putText(docu_img, "heat_max: "+str(heat_max), (100, 250), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(folder+'/all/all_'+str(nnn)+'.jpg', docu_img)
#    '''
    
    # label each frame with a counter for debugging / threshold
    cv2.putText(draw_image, str(nnn) , (50, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(draw_image, "hot windows found: "+str(len(hot_windows)) , (150, 100), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite(folder+'/output/out_'+str(nnn)+'.jpg', draw_image)
	
    return draw_image



#videoname = 'test_video_test.mp4' labels
videoname = folder+'/project_video_processed.mp4' 
output = videoname
clip1 = VideoFileClip("input_video/project_video.mp4").subclip(substart, subend)
clip = clip1.fl_image(videopipe) #NOTE: this function expects color images!!
clip.write_videofile(output, audio=False)



