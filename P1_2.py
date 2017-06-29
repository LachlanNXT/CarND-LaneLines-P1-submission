#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, a=0.8, b=1., c=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * a + img * b + c
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, a, img, b, c)
	
import os
import time

test_list = os.listdir("test_images/")

def displayim(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

challenge = False
def pipeline(image_name, write_out = False, display = False):
    # Read in and grayscale the image
    if write_out:
        image = cv2.imread("test_images/" + image_name, cv2.COLOR_RGB2HSV)
        try:
            os.mkdir('test_images_output_2')
        except:
            pass
    else:
        image = image_name
    if display:
        cv2.imshow('image',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()  
    
    #convert to grayscale
    gray = image[:,:,2]#cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    if display:
        cv2.imshow('image',gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
       
    '''
    pts_src and pts_dst are numpy arrays of points
    in source and destination images. We need at least 
    4 corresponding points. 
    inputQuad[0] = Point2f( 0,0);
        inputQuad[1] = Point2f( IPM_ROI.cols,0);
        inputQuad[2] = Point2f( IPM_ROI.cols,IPM_ROI.rows);
        inputQuad[3] = Point2f( 0,IPM_ROI.rows);           //
        // The 4 points where the mapping is to be done , from top-left in clockwise order
        outputQuad[0] = Point2f( 0,0 );
        outputQuad[1] = Point2f( mFrame.cols,0);
        outputQuad[2] = Point2f( mFrame.cols-250,mFrame.rows);
        outputQuad[3] = Point2f( 250,mFrame.rows);
    '''
    x = gray.shape[1]; y = gray.shape[0];
    pts_src = np.array([[(0,y*.58),(x,y*.58),(x, y), (0,y)]], dtype=np.int32)
    pts_dst = np.array([[(0,0),(x,0),(x-200, y), (200,y)]], dtype=np.int32)
    h, status = cv2.findHomography(pts_src, pts_dst)

    ''' 
    The calculated homography can be used to warp 
    the source image to destination. Size is the 
    size (width,height) of im_dst
    '''
    size = (gray.shape[1],gray.shape[0])
    im_dst = cv2.warpPerspective(gray, h, size)

    if display:
        cv2.imshow('image',im_dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(im_dst,(kernel_size, kernel_size),0)

    if display:
        cv2.imshow('image',blur_gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Define our parameters for Canny and apply
    # compute the median of the single channel pixel intensities
    sigma=0.33
    v = np.median(blur_gray)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    #print(lower,upper)
    
    edges = cv2.Canny(blur_gray, lower, upper)

    if display:
        cv2.imshow('image', edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
    
    # Create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(gray)   
    ignore_mask_color = 255

    if challenge:
        vertices = np.array([[(200,y-100),(600, 0), (x-600, 0), (x-200,y-100)]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked = cv2.bitwise_and(edges, mask)
        #return masked
    else:
        # This time we are defining a four sided polygon to mask
        vertices = np.array([[(200,y),(450, 0), (x-450, 0), (x-200,y)]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked = cv2.bitwise_and(edges, mask)
        
    if display:
        cv2.imshow('image',masked)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/90 # angular resolution in radians of the Hough grid
    threshold = 60     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 200 #minimum number of pixels making up a line
    max_line_gap = 200    # maximum gap in pixels between connectable line segments
    line_image = np.copy(image)*0 # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(masked, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)

    line_img = hough_lines(masked, rho, theta, threshold,
                                min_line_length, max_line_gap)
    
    if display:
        cv2.imshow('image', line_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Categorise lines as left lane or right lane and find the average parameters
    #plt.imshow(line_image)
    neg = []
    neg_c = []
    pos = []
    pos_c = []
    height = y
    for line in lines:
        for x1,y1,x2,y2 in line:
            gradient = ((y2-y1)/(x2-x1))
            if math.isnan(gradient): gradient = 999;
            if gradient < -2 and gradient >-5:
                neg.append(gradient)
                neg_c.append(y1-gradient*x1)
            if gradient > 2 and gradient < 5:
                pos.append(gradient)
                pos_c.append(y1-gradient*x1)
            if height>y2: height=y2;
            if height>y1: height=y1;
    
    # Left lane line
    if len(pos)>0:
        left = np.mean(pos)
        left_c = np.mean(pos_c)
        ly1 = y
        lx1 = int((ly1 - left_c)/left)
        ly2 = height
        lx2 = int((ly2 - left_c)/left)
        left_line = [[lx1,ly1,lx2,ly2]]
    else:
        left_line = [[0,0,0,0]]
        
    # Right lane line
    if len(neg)>0:
        right = np.mean(neg)
        right_c = np.mean(neg_c)
        ry1 = y
        rx1 = int((ry1 - right_c)/right)
        ry2 = height
        rx2 = int((ry2 - right_c)/right)
        right_line = [[rx1,ry1,rx2,ry2]]
    else:
        right_line = [[0,0,0,0]]
        
    # Line image to merge with original image
    final_lines = [left_line,right_line]
    #print(final_lines)
    draw_lines(line_image, final_lines, [255, 0, 0], 10)

    if display:
        cv2.imshow('image', line_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    hdash = np.linalg.inv(h)
    lines_original = cv2.warpPerspective(line_image, hdash, size)
    
    if display:
        cv2.imshow('image', lines_original)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Draw the lines on the edge image
    lines_overlay = cv2.addWeighted(image, 0.8, lines_original, 1, 0) 
    #plt.imshow(lines_edges)
    
    if display:
        cv2.imshow('image', lines_overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    
    # Save images
    if write_out:
        cv2.imwrite(os.path.join('test_images_output_2',image_name), cv2.cvtColor(lines_overlay, cv2.COLOR_RGB2BGR))
        
    return lines_overlay

# [pipeline(x, True) for x in test_list];

# cv2.imshow('image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    result = pipeline(image)

    return result

white_output = 'test_videos_output_2/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
try:
    os.mkdir('test_videos_output_2')
except:
    pass
white_clip.write_videofile(white_output, audio=False)


yellow_output = 'test_videos_output_2/solidYellowLeft.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
yellow_clip.write_videofile(yellow_output, audio=False)

challenge_output = 'test_videos_output_2/challenge.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
challenge = True
clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
challenge_clip.write_videofile(challenge_output, audio=False)

