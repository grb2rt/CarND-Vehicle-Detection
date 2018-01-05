
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from scipy.ndimage.measurements import label



def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
#        if ((box[0][1]-box[1][1])*(box[1][0]-box[0][0]))>4096: # min detection 64x64px
#            print(str((box[0][1]-box[1][1])*(box[1][0]-box[0][0])))
#        print(box)
        x = (box[1][1], box[0][1])
        y = (box[0][0], box[1][0])
        maxix = np.amax(x)
        minix = np.amin(x)
        maxiy = np.amax(y)
        miniy = np.amin(y)

        heatmap[minix:maxix, miniy:maxiy] += 1
#        heatmap[box[1][1]:box[0][1], box[0][0]:box[1][0]] += 1

##      test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
#       test_img = cv2.resize(img[window[1][1]:window[0][1], window[0][0]:window[1][0]], (64, 64))      

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    bbox_list = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        color = (0,0,0)
        if car_number ==1:
            color = (0,255,255)		
        if car_number ==2:
            color = (255,255,0)		
        if car_number ==3:
            color = (255,0,255)		
#        else:
#            color = (0,0,255)
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
#        print((bbox[1][1]-bbox[0][1])*(bbox[1][0]-bbox[0][0]))
        if ((bbox[1][1]-bbox[0][1])*(bbox[1][0]-bbox[0][0]))>1600: # min detection 40x40px
#            print(str((box[0][1]-box[1][1])*(box[1][0]-box[0][0])))
            cv2.rectangle(img, bbox[0], bbox[1], color, 6)
            bbox_list.append(bbox)
    # Return the image
    return img, bbox_list

def heathot(hot_windows, test_image, dyn_threshold, abs_threshold, max_threshold):

    box_list = hot_windows
    # create blank image 
    heat = np.zeros_like(test_image[:,:,0]).astype(np.float)
    
    # Add heat to each box in box list
    heat = add_heat(heat,box_list)
    global heat_max
    heat_max = np.amax(heat)
#    print("run: ")
#    print("heat max = "+str(heat_max))
#    print("hot windows: "+str(hot_windows))
#    print('heat_max = ' + heat_max)
	
    # Apply dynamic threshold to help remove false positives
    if dyn_threshold <= abs_threshold :
        heat = apply_threshold(heat, abs_threshold)	# basically return empty heatmap
#        print('too low: abs_threshold= '+str(abs_threshold)+' dyn_threshold= '+str(dyn_threshold))
    if dyn_threshold > abs_threshold and dyn_threshold <= max_threshold:
        heat = apply_threshold(heat, dyn_threshold)
#        print('middle detected= '+str(dyn_threshold))
#        print(str(np.amax(heat)))
    if dyn_threshold > max_threshold :
        heat = apply_threshold(heat, max_threshold)
#        print('too high')
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
#    print('labels shape = '+labels.shape)
	
    return labels, heat_max
