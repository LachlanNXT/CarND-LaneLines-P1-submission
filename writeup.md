# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./test_images_output/solidWhiteCurve.jpg "out1"
[image2]: ./test_images_output/solidWhiteRight.jpg "out2"
[image3]: ./test_images_output/solidYellowCurve.jpg "out3"
[image4]: ./test_images_output/solidYellowCurve2.jpg "out4"
[image5]: ./test_images_output/solidYellowLeft.jpg "out5"
[image6]: ./test_images_output/whiteCarLaneSwitch.jpg "out6"

---

### Reflection

### 1. Describe your pipeline.

The image processing pipeline consisted of these steps: 
* convert to HSV and colour threshold for yellow and white
* convert to grayscale
* apply Gaussian smoothing
* edge detection
* image masking
* Hough transform
* filter lines to two lane lines

HSV is less affected by ambient brightness than RGB. This means it is a little more robust to shadows etc. I could improve this further by using chromaticity but the test videos are not that demanding and there does not seem to be an opencv built in function for this conversion, so I'd have to write it in python and that would be a bit slow.

Convert to grayscale so we can do edge detection and Hough transform, smoothing removes some noise.

For edge detection I use the method described here to automatically calculate parameters for the Canny edge detector:
http://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
It seems fairly robust, and saves the time tuning parameters manually.

Image masking is performed so the Hough transform doesn't pick up lines I don't want that aren't part of the road.

Hough transform finds lines in a grayscale image. I used a slightly larger than default grid size, because I don't think the angles of the road lines need to be any better than 2 degrees and it's speedier that way. I fiddled around with the parameters a bit until I found good ones that pick up lines well.

The post processing basically involves categorising the Hough lines into left lane lines, right lane lines, or discard (noise). This was done by exploiting the known range of angles that the lines would appear in the video given that on the highway those angles don't vary much. I calculate the average gradient and intercept of all the lines corresponding to each lane, then draw the average line for each lane from the bottom of the image to the highest point on the image a line was found (within the masked area corresponding to the road).

![alt text][image1] ![alt text][image2] ![alt text][image3]
![alt text][image4] ![alt text][image5] ![alt text][image6]

### 2. Identify potential shortcomings with your current pipeline

You could come up with a very long list of shortcomings with the methods used here, but they all fall under the more general problem of being too problem-specific. I would assert that the colour selection, edge detection, Hough transform parameters and post-processing are not robust to a wide range of driving conditions. Weather, brightness, etc. could prevent the image processing from working properly. Measures have been taken to address this somewhat, but it would not be good enough to venture far from perfect conditions.

Additionally, probably the biggest shortcoming is that this method only works when the road is relatively straight and not busy, because it searches for straight lines. If the car went around a corner or there were obstructions so not much line was visible I'm sure this would not work at all. So it is basically good for highway driving without much traffic in good weather, which is a very limited subset of the operational domain of a true self-driving car.

Finally, I'm not sure if this runs at real time speed which is obviously important.

### 3. Suggest possible improvements to your pipeline

More finely tune parameters, or use totally different methods (deep learning?) - I'm in favour of the latter.
