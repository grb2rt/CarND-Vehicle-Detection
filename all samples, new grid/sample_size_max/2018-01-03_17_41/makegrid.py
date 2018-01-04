
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import lesson_functions
#import makegrid
import detect


def makegrid():
    
    img_max = 1279 # image size

    # these are the starting points for all lower, left corners of rectangles
    x_min = 527
    y_max = 640
    y_min = 440

    # some variables to generate the grid    
    kernel_start = 64
    kernel_step = 8
#    step = 20
    step = 8
    rows = 2
    
	# list for results
    bboxes = [] # here are all the rectangles
	
    for z in range(0,26,1):
        kernel = kernel_start + z*kernel_step
        x_min_base = x_min + z*2*step 
        y_min_base = y_min + z*step
        		
        for n in range(0,rows,1):
            y = y_min_base + n*step
            x_min_tmp = x_min_base + n*step*2
#            print()
#            print("first position in row = x: "+str(x_min_tmp)+" y: "+str(y))
#            print("current kernel: "+str(kernel))
            for m in range(0, int((img_max-kernel-x_min_tmp)/step)+1, 1):
                x = m*step + x_min_tmp
                if y <= y_max:
                    bboxes.append(((x,y),(x+kernel, y-kernel)))
#            print("last saved x = "+str(x)+" y: "+str(y))
#            print("last position x in row = "+str(x)+" y: "+str(y))
#            print("no positions in row: "+str(int((img_max-kernel-x_min_tmp)/step)+1))
#        print("length of block "+str(z)+" = "+str(len(bboxes)))
#        print()
#        print()
    return bboxes

bboxes = makegrid()
			
#print("initial_loop_x: " + str(initial_loop_x))			
#print("initial_loop_y: "+str(initial_loop_y))			
#print("bboxes length: "+str(len(bboxes)))		
#print("bboxes: ")
#print(bboxes)			
# in total 6812 frames



##############################################################
# import from udacity  --> just for debugging & documentation
##############################################################


image = mpimg.imread('../test_images/test1.jpg')

# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy
    
    
windows = makegrid()
                   

imagez = glob.glob('../test_images/test*.jpg')
for fname in imagez:
    img = cv2.imread(fname)
    file = fname[15:]
    window_img = draw_boxes(img, windows, color=(0, 0, 255), thick=2)                    
    cv2.imwrite('../docu/makegrid_'+file, window_img)
cv2.destroyAllWindows()

