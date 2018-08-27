# The following program automates analysis of fruit flies by tracking the motion of its proboscis. The distance and time
# taken of the extension is recorded as well as the coordinates of motion with different approaches.

from __future__ import division
import os 
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline, interp1d, splrep, splev, interp2d, Rbf
import cv2
import math
import csv
import imutils
from PIL import Image
import argparse

# Define the video location as File and the folder that the video is in as FileDir. These are used later on to read,
# open, save the video and images.

FileDir ="E:\\Flies\\24.07.2017\\FLY10\\"
File = "E:\\Flies\\24.07.2017\\FLY10\\FLY_18_50ms_100ms_2017-07-24-150533-0000.avi"

# Appending farsight toolkit to the system path so they could be used.
# Use print sys.path to find directories in path and append as necessary.

sys.path.append("C:\\ProgramData\\Anaconda2\\Lib\\site-packages\\farsight\\python")
sys.path.append("C:\\ProgramData\\Anaconda2\\Lib\\site-packages\\farsight\\python\\Common")
sys.path.append("C:\\ProgramData\\Anaconda2\\Lib\\site-packages\\farsight\\bin")
sys.path.append("C:\\ProgramData\\Anaconda2\\Lib\\site-packages\\farsight\\python\\XML")
sys.path.append("C:\\ProgramData\\Anaconda2\\Lib\\site-packages\\video-analysis-master\\video")
sys.path.append("C:\\ProgramData\\Anaconda2\\Lib\\site-packages\\video-analysis-master\\external")
sys.path.append("C:\\ProgramData\\Anaconda2\\Lib\\site-packages\\video-analysis-master\\video\\gui")
sys.path.append("C:\\ProgramData\\Anaconda2\\Lib\\site-packages\\video-analysis-master\\video\\io")
sys.path.append("C:\\ProgramData\\Anaconda2\\Lib\\site-packages\\video-analysis-master\\video\\analysis")
sys.path.append("C:\\ProgramData\\Anaconda2\\Lib\\site-packages\\py-utils-master\\utils")
sys.path.append("C:\\ProgramData\\Anaconda2\\pkgs\\itk-4.12.0-vc9_0\\Library")
sys.path.append("C:\ProgramData\Anaconda2\Lib\site-packages\py-utils-master\utils\data_structures")
sys.path.append("C:\ProgramData\Anaconda2\Lib\site-packages\py-utils-master\utils\geometry")
sys.path.append("C:\ProgramData\Anaconda2\Lib\site-packages\py-utils-master\utils\link")
sys.path.append("C:\ProgramData\Anaconda2\Lib\site-packages\py-utils-master\utils\math")
sys.path.append("C:\ProgramData\Anaconda2\Lib\site-packages\py-utils-master\utils\numba")
sys.path.append("C:\ProgramData\Anaconda2\Lib\site-packages\py-utils-master\utils\plotting")
sys.path.append("C:\ProgramData\Anaconda2\Lib\site-packages\py-utils-master\utils\tests")
sys.path.append("C:\ProgramData\Anaconda2\Lib\site-packages\py-utils-master\utils")

from curves import point_distance, angle_between_points
from image import subpixels, get_subimage, set_image_border, mask_thinning, detect_peaks, regionprops, get_steepest_point
from filters import FilterFunction, FilterCrop, FilterOpticalFlow
import points

# use subprocess module to call farsight executables from the bin folder
#subprocess.call(["C:\\ProgramData\\Anaconda2\\Lib\\site-packages\\farsight\\bin\\prep.exe"])

import farsightutils

# Set a work directory for the current session.

WorkDir = farsightutils.SetWorkingDirectory()

# Shows all the files within the working directory and opencv version.

#print "Files in the working directory are: ", farsightutils.ls(), "\n"
#print "openCV version is: ", cv2.__version__,"\n"

# Setting the name of the window that can be used later.

winName = "FLY_5_50ms_100ms_2017-07-27-151901-0000"

# A function for counting the total number of frames in the video.

def FrameCounter(path):
    video = cv2.VideoCapture(path,0)
    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    return total

nof = FrameCounter(File)
print "Total number of frames in video is: ", nof,"\n"
print "Length of video in seconds: ", float(nof)/163.317, "sec", "\n"

# Following function saves all the frames in the video as .png files.

def VideoToFrames(vid):
    
# Count is used to number the frames. Starting from 100 because there are usually between 100 and 1000 frames and
# this makes sorting easier later.

    count = 100
    frames = []

# Function to get the line shape of each frame if interested (didn't work too well as too much noise).

    def lines(image, sigma):
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        """blurred = cv2.bilateralFilter(image,9,75,75)"""
        
# Compute the median of the single channel pixel intensities.

        v = np.median(blurred)
     
# Apply automatic Canny edge detection using the computed median.

        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(blurred, lower, upper)
        """peaks = np.asarray(detect_peaks(edged)) 
        cent = regionprops.centroid(edged)
        img = mask_thinning(edged[peaks])"""

# Return the edged image.

        return edged
    
    if (vid.isOpened() == False):
        print "Error opening video file"
    while True:
        ret,frame = vid.read()
    
        cv2.imwrite("Frame%d.png"%count,frame)#lines(frame,0.01))
        """imagePath = FileDir + "Frame%d.png"%count

        #load the image, convert it to grayscale, and blur it slightly
        image = cv2.imread(imagePath)
        for imagePath in glob.glob(FileDir + "/*.png"):
            image = cv2.imread(imagePath)
            blurred = cv2.GaussianBlur(image, (3, 3), 0)
            cv2.imshow("blur",blurred)
        	  apply Canny edge detection using a wide threshold, tight
              threshold, and automatically determined threshold
            wide = cv2.Canny(blurred, 10, 200)
            tight = cv2.Canny(blurred, 225, 250)
            auto = lines(blurred,0.33)
            new_im = np.hstack(auto)
            cv2.imshow("lines",new_im)
            cv2.imwrite("Frame%d.jpg"%count,new_im)"""
        frames.append(count)
        count = count + 1
        if count >= nof:
            break
        
# Returns all the ending values of the saved images (i.e. 100, 101, 102....) as an array that can be looped later.

    framess = np.asarray(frames)
    return framess

# This function looks for the frame at which the light flash has ended and deletes all frames
# before that as they are not of interest.

def FrameDelete(count):
    pixels = []
    counts = np.asarray(count)
    for c in count:
    
        im = Image.open(FileDir+"Frame%d.png"%c)
        arr = np.asarray(im)
        mn = arr.mean(-1)
        total = arr.sum(0).sum(0)
        pixels.append(sum(total))
        if pixels[c-100] + 75000000 < pixels[c-101] and pixels[c-100] - 10000000 < pixels[c-99]:
            i = count[0]
            while i < c:
                os.remove(FileDir+"Frame%d.png"%i)
                np.delete(counts,i-100,0)
                i = i + 1
            
            i = i + 250
            if i < len(count):
                while i < len(count):
                    os.remove((FileDir)+"Frame%d.png"%i)
                    np.delete(counts,i-100,0)
                    i = i + 1
            break
    return counts

# Function to write a new video from individual frames, takes the format (avi or mp4 for example) and name as arguments.

def WriteNewVideo(ext,winName):
    
    # Construct the argument parser and parse the arguments
    
    ap = argparse.ArgumentParser()
    
    #ap.add_argument("-ext", "--extension", required=False, default='jpg', help="extension name. default is 'jpg'.")
    
    ap.add_argument("-o", "--output", required=False, default=winName+".avi", help="output video file")
    args = vars(ap.parse_args())
    # Arguments
    
    dir_path = FileDir
    ext = '.'+ ext
    output = args['output']

# Takes images from the working directory and sorts them in ascending order.

    images = []
    for f in os.listdir(dir_path):
        if f.endswith(ext):
            images.append(f)
    images.sort()

# Determines the width and height from the first image

    image_path = os.path.join(dir_path, images[0])
    frame = cv2.imread(image_path)
    #cv2.imshow('video',frame)
    height, width, channels = frame.shape
    
# Define the codec and create VideoWriter object

    fourcc = cv2.VideoWriter_fourcc(*'XVID') # Be sure to use lower case
    out = cv2.VideoWriter(output, fourcc, 15, (width, height))
    c = 0
    for image in images:
    
        image_path = os.path.join(dir_path, image)
        frame = cv2.imread(image_path)
    
        out.write(frame) # Write out frame to video
        c = c+1
        if c > 250:
            break
        #cv2.imshow('video',frame)
        #if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
         #   break
    
    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()
    
    print("The output video is {}".format(output))
    return FileDir + output

# Function for rounding calculations to certain significant figures (2 by default).

def round_sig(x, sig=2):
    return round(x, sig-int(math.floor(math.log10(abs(x))))-1)

"""vid = cv2.VideoCapture(File)
newimages = VideoToFrames(vid)
count = FrameDelete(newimages)
newvid = WriteNewVideo("png", "FLY_12_new")
vid = cv2.VideoCapture(newvid)
ret,frame = vid.read()"""

# Function to determine the frame at which the flash has exactly passed so that the proboscis extension can be followed.
# Function takes the video and the number of its frames as arguments.

def frameposition(vid,nof):
    c = 0
    pixels = []
    nof = int(nof)
    counts = np.linspace(1,nof,nof)
    
# Looping over all the frames and summing the pixel values of each frame.

    while True and c < nof+1:
        ret,frame = vid.read()
        
        im = frame
        arr = np.asarray(im)
        total = arr.sum(0).sum(0)
        pixels.append(sum(total))
        
# If the total pixel values is much less than on the previous frame and also lower than a certain value (which it should
# be if there is no illumination) then the frame is saved as a png image and its position saved and the loop stops.

        if pixels[int(c)] + 80000000 < pixels[int(c-1)] and pixels[int(c)] < 300000000  :
            pos = int(c)
            vid.set(1,pos)
            cv2.imwrite("1stFrame.png",frame)
            print "success"
            vid.release()
            cv2.destroyAllWindows
            break
        c = c+1
    return pos

# Sorting all the images in the folder in the correct sequence if using image saving method.

images = []
for f in os.listdir(FileDir):
    if f.endswith(".png"):
        images.append(FileDir+f)
        
images.sort()
start = frameposition(cv2.VideoCapture(File),nof)
print start
# Function to select the region of interest. Enables to crop out the proboscis so there would be less
# interference. The rectangle expands from the centre and when selected it is saved and "esc" key has to be used to
# close the window and move on.

if True:
    fromCenter = False
    showCrossair = False
    Firstim = cv2.imread(FileDir + "1stFrame.png")#cv2.imread(frameposition(vid,nof)[1])
    r = 1200./Firstim.shape[1]
    dim = (1200, int(Firstim.shape[0]*r))
    Firstim = cv2.resize(Firstim,dim,cv2.INTER_LINEAR)
    roi = cv2.selectROI("selectROI",Firstim)
    Crop = Firstim[int(roi[1]):int(roi[1]+roi[3]),int(roi[0]):int(roi[0]+roi[2])]
    height,width = Crop.shape[:2]
    print width,height
    cv2.waitKey(27)
cv2.destroyAllWindows()

# Parameters that influence the masking process of the motion. 

if True:
    KERN_SIZE = 15
    RADIUS = 5
    THRESHOLD_AT = 25
    INPUT_SIZE_THRESHOLD = 25
    MINIMUM_PATH_SIZE = 5
    MINIMUM_PATH_STD = 3*RADIUS
    WRITE_TO_FILE = False

# Various parameters and functions necessary to save a new video.

video_outputfile = "default_output.avi"

def avgit(x):
    return x.sum(axis=0)/np.shape(x)[0]
def plotp(p,mat,color=255):
    mat[p[0,1],p[0,0]] = color
    return mat[p[0,1],p[0,0]]
if len(sys.argv) != 3:
    cap = cv2.VideoCapture(File)
    
else:
    cap = cv2.VideoCapture(sys.argv[1],0)
    video_outputfile = sys.argv[2]
"""fourcc = cv2.VideoWriter_fourcc(*'XVID')
ret, frame = cap.read()
height, width, layers = frame.shape
video_out = cv2.VideoWriter(video_outputfile, fourcc, 15, (width, height), True)
print video_out.isOpened()"""

# Defining the background removal functions (another one to use is MOG2)

fgbg = cv2.createBackgroundSubtractorKNN()
bwsub= cv2.createBackgroundSubtractorKNN()

kernlen = KERN_SIZE
kern = np.ones((kernlen,kernlen))/(kernlen**2)
ddepth = -1

# Defining the blurring functions for the video.

def blur(image):
    return cv2.filter2D(image,ddepth,kern)
def blr_thr(image, val=133):
    return cv2.threshold(blur(image),val,255,cv2.THRESH_BINARY)[1]

# Normalize function for video saving.
def normalize(image):
    s = np.sum(image)
    if s == 0:
       return image
    return height*width* image / s

# Renaming the parameters defined earlier.

rad = RADIUS
thresh_at = THRESHOLD_AT
THIS_MUCH_IS_NOISE = INPUT_SIZE_THRESHOLD

# Creating empty arrays to store various parameters of the proboscis extension.

paths = []
archive = []
Bottom_x=[]
Bottom_y=[]
Bot_dist = []
Centroid_x = []
Centroid_y = []
Mid_dist = []
Ellipse_mid_x = []
Ellipse_mid_y = []
Ellipse_mid_dist = []
Ellipse_frames = []
Frame_count = []
Frame_count2 = []
Frame_count_mid = []
pixels = []
angles = []
Furthest = []
Distances = []
Moredist = []
Farx = []
Fary = []
x = []
y = []
Botx = []
Boty = []
# i is used to count the number of frames as the video is read.

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter('Output.avi',-1,15.0,(width,height))

i = 0
d = 0 

# Reading the video (and resizing the video so that it fits my screen). "cv2.imread" and "while True" is used when working
# with images instead of video.

#while True:
while cap.isOpened():
    ret,frame = cap.read()
    #frame = cv2.imread(images[i])
    r = 1200./frame.shape[1]
    dim = (1200, int(frame.shape[0]*r))
    frame = cv2.resize(frame,dim,cv2.INTER_LINEAR)

# Cropping the video to the region of interest defined earlier with the blue rectangle.

    Crop = frame[int(roi[1]):int(roi[1]+roi[3]),int(roi[0]):int(roi[0]+roi[2])]

# Converting the video to grayscale just in case.

    gray = cv2.cvtColor(Crop, cv2.COLOR_RGB2GRAY)

# Applying background subtractor (fgbg defined earlier) to the cropped video and blurring the video to make it smoother.

    fgmask = fgbg.apply(Crop)
    mask = blur(fgmask)

# Thresholding the video to draw the contour lines around the moving object. Using opencv threshold function and
# the erode and dilate functions for finding the most bottom point as these makes the mask more consistent.  
 
    ret2, thresh = cv2.threshold(mask, thresh_at, 255, cv2.THRESH_BINARY_INV) 
    thresh2 = cv2.erode(thresh, None, iterations=1)
    thresh2 = cv2.dilate(mask, None, iterations=1)

# Getting the contours of the thersholded objqect.

    img2, cons, hierarchy = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cntz = cv2.findContours(thresh2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cntz[0] if imutils.is_cv2() else cntz[1]
    
    con = np.asarray(cons)
# Finding the moments of the contour to calculate the centroid of the mask.

    M = cv2.moments(img2)
    
# Starting tracking when the contours are detected and the flash has passed (i >= start, where start is the first
# frame after the flash as defined earlier).

    if cnts != [] and i > start:
        
# Finding the max values of the contour i.e. the outline points of the mask.
       # if hierarchy[0][1] != []:
            
        c = cnts[0] #max(cnts , key = cv2.contourArea)
        min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(img2,mask = thresh2)
# Defining the most bottom extreme point of the mask

        bottom = tuple(c[c[:, :, 1].argmax()][0])
        Botx.append(bottom[0])
        Boty.append(bottom[1])
# Appending the most bottom point coordinates after each iteration and calculating the actual distance moved.
# Also recording the passing frames to compare against time.
        Bottom_x.append(bottom[0])
        Bottom_y.append(bottom[1])
        bot_distance = 5*5.86*point_distance((Bottom_x[0],Bottom_y[0]),bottom)
        Bot_dist.append(bot_distance)
        Frame_count.append(i)
       
#==============================================================================
#==============================================================================
        #cv2.circle(gray,bottom,5,(255,255,255),-1)
#==============================================================================
#==============================================================================
# Fitting an eclipse around the contour points or the mask.

    for c in cons:
       
# Fitting only starts if there are 5 or more outline points as opencv requires this amount for fitting. Also
# fitting starts only after the flash has passed.

        if len(c) > 5 and i > start:
            fitellipse = cv2.fitEllipse(c)
            ellipse = cv2.ellipse(img2,fitellipse,(0,0,0),1)

# Calculating the moments of the ellipse and its centroid. Recording the centroid coordinates after each iteration
# and calculating the distance of its motion as well as all the frames that this occurs for (The number of frames
# may be different because there aren't always 5 or more points of the outline)

            Me = cv2.moments(ellipse)
            Ex = int(Me['m10']/Me['m00'])
            Ey = int(Me['m01']/Me['m00'])
            Ellipse_mid_x.append(Ex)
            Ellipse_mid_y.append(Ey)
            Ellipse_mid_distance = 5*5.86*point_distance((Ellipse_mid_x[0],Ellipse_mid_y[0]),(Ex,Ey))
            Ellipse_mid_dist.append(Ellipse_mid_distance)
            Ellipse_frames.append(i)

# Calculating the centroid of the mask if there aren't 0 values for division and the flash has passed.

    if M['m00'] != 0 and i > start:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

# Recording the centroid coordinates after each iteration and calculating the distance from its initial position.

        Centroid_x.append(cx)
        Centroid_y.append(cy)
        

# Mid_0 saves the initial position of the centroid and this acts as the "0 position".
        
        Mid_0 = (Centroid_x[0],Centroid_y[0])
        mid_distance = 5*5.86*point_distance(Mid_0,(cx,cy))
        Mid_dist.append(mid_distance)
        Frame_count_mid.append(i)

# if statement is necessary when the ellipse centroid is used as len(pts) not always greater than 5.

        if cnts != []:
            pts = cnts[0]

# Finding the distance of each outline point from the centroid and the angle formed by three points:
# the most bottom point, the centroid and every point of the outline (separately).
    
            if True:
                for pt in pts:
                    pt = np.transpose(pt)
                    distance = point_distance((cx,cy),(int(pt[0]),int(pt[1])))
                    angle = angle_between_points(bottom,(cx,cy),(int(pt[0]),int(pt[1])))
                    
                    angles.append(angle)
                    
# Recording the distance for all points in order to determine the necessary coordinate later on.
    
                    Moredist.append(distance)
                    
# Recording the distances that lie below the centroid and at an angle less than 45 degrees because we do 

# not want points that are close to the mouth by accident.
                    
                    if Moredist != [] and  int(pt[1]) > int(Mid_0[1]) and int(pt[0]) > 0.7*int(cx) and int(pt[0]) < 2*int(cx):
                        if d >= 3:
                            if int(pt[1]) > 1.5*int(Ey) and point_distance((Farx[d-1],Fary[d-1]),(int(pt[0]),int(pt[1]))) < 1.5*point_distance((Farx[d-2],Fary[d-2]),(int(pt[0]),int(pt[1]))):
                                Distances.append(distance)
                        
                                #and 5*5.86*point_distance((Farx[d-1],Fary[d-1]),(pt[0],pt[1])) < 400:
                       
                        elif d <= 3:
                            Distances.append(distance)

# If any distances below the centroid were recorded then the maximum of them will be recorded and its coordinates
# as well. The circle function can be used to confirm the position of the point.
    
                if Distances != []:
                    
                    D = max(Distances)
                    Frame_count2.append(i)
                    Farx.append(np.transpose(pts[Moredist.index(D)])[0])
                    Fary.append(np.transpose(pts[Moredist.index(D)])[1])
                    Furthest_0 = (Farx[0],Fary[0])
                    Furthest.append(5*5.86*max(Distances))
                    #cv2.circle(gray,(cx,cy),5,(255,255,255),-1)
                    cv2.circle(gray,(np.transpose(pts[Moredist.index(D)])[0],np.transpose(pts[Moredist.index(D)])[1]),5,(255,255,255),-1)
# The distance of this iteration points are cleared for the next iteration (since the mask changes due to motion).
                    d = d + 1
                    Distances = []
                Moredist = []

# If there is no points or division by 0, only 0s will be the centroid coordinates but these are not recorded.

    else:
        cx,cy = 0,0
     
# Filtering the image and removing noise, recording the path and overlaying the mask on the image.

    scatter = map(avgit, cons)
    filterWith = lambda x: len(x) > MINIMUM_PATH_SIZE and np.std(x) > MINIMUM_PATH_STD
    (toArchive, paths) = points.extendPaths(rad, paths, scatter, filterWith, noisy=(len(scatter) > THIS_MUCH_IS_NOISE), discard=False)
    archive += toArchive
    img = (1-img2)*gray

#Plotting the mask path using the plotp function defined earlier and the polylines function of opencv.

    for path in archive:
        color = 0
        cv2.polylines(img, np.int32([reduce(lambda x,y: np.append(x,y,axis=0), path)]), 0, (255,0,0))
        for pnt in path:
            plotp(pnt, img, color=color)
            
    for path in paths:
        color = 255
        cv2.polylines(img, np.int32([reduce(lambda x,y: np.append(x,y,axis=0), path)]), 1, (0,0,125))
        for pnt in path:
            line = plotp(pnt, img, color=color)

# Showing the video.
    out.write(img)
    cv2.imshow('frame', img)
    
# Recording stops and video closes if "q" is pressed.
    
    if cv2.waitKey(1) & 0xFF == ord('q') or i == nof - 1:
        break
    i = i + 1
cap.release()
out.release()
cv2.destroyAllWindows()

# Function that takes x and y values of a curve and normalises the curve. Returns normalised x and y values as numpy
# arrays.

def Normalize_curve(x,y):
    npntsx = []
    npntsy = []
    for i in range(len(x)):
        xs = (x[i] - min(x))/(max(x)-min(x))
        ys = (y[i] - min(y))/(max(y)-min(y))
        npntsx.append(xs)
        npntsy.append(ys)
    
    l = len(npntsy)
    print "Number of points is: ", l, "\n"
    return npntsx,npntsy

def residual(y,fit):
    i = 0                               # Initial value in the array of x values.
    S = 0                               # Initial residual of 0.
    while i<len(fit):                     # Loops over all the values in the array of x values.
            
                    # Calculates the residual for each data point and adds them up.
        
        S = S+(fit[i] - y[i])**2
        i = i+10
    return S

# Plotting figures using the various parameters calculated earlier.
# Smooth_curve function is used to smooth the curve by interpolating the points. By default cubic splines are
# used with the number of support points is the number of points supplied.

plt.figure(1)

(tck1),fp1,ier1,msg1 = splrep(Frame_count2, Furthest, s=10, k=3, full_output = True)
Framecountnew = np.arange(Frame_count2[0], max(Frame_count2), 0.1)
Framecounttest = np.arange(Frame_count2[0], max(Frame_count2), 0.2)
Botdistnew = splev(Framecountnew,tck1,der=0)
Botdisttest = splev(Framecounttest,tck1,der=0)
print len(Frame_count2), len(np.asarray(Fary))

#plt.plot(Framecountnew,Botdistnew,'-')
plt.plot(Framecountnew,Botdistnew,'-')
plt.plot(Frame_count2,Furthest,'.')
plt.ylabel("Extension distance (microns)")
plt.xlabel("Time (frames)")
plt.title("Figure 1. Distance of the furthest point of the proboscis")
plt.savefig("Figure_1_Distance_of_the_lowest_point_of_the_proboscis.png")

plt.figure(2)
plt.plot(Frame_count2, Furthest, '.')
plt.plot(Frame_count2, Furthest, '-')
plt.ylabel("Furthest point from centroid")
plt.xlabel("Time (frames)")
plt.title("Figure 2. Distance of the furthest point from the centroid with no smoothing")
plt.savefig("Figure_2_Furthest_point_from_centroid_distance_vs_time.png")

znew = np.arange(Frame_count2[0], max(Frame_count2), 0.1)
(tck2), fp2, ier2, msg2 = splrep(Frame_count2, Farx, s=10, k=3, full_output = True)
(tck3), fp3, ier3, msg3 = splrep(Frame_count2, Fary, s=10, k=3, full_output = True)
Botxnew = splev(znew,tck2,der=0)
Botynew = splev(znew,tck3,der=0)

plt.figure(3)
plt.plot(Farx, Fary,'.')
plt.plot(Botxnew, Botynew,'-')
plt.ylabel("Furthest point extension y")
plt.xlabel("Furthest point extension x")
plt.title("Coordinates of the furthest point motion")
plt.title("Figure 3. Furthest point from the centroid coordinates")
plt.gca().invert_yaxis()
plt.savefig("Figure_3_Furthest_point_from_centroid_coordinates.png")

plt.figure(4)
plt.plot(Normalize_curve(Farx,Fary)[0],Normalize_curve(Farx,Fary)[1],'.')
plt.plot(Normalize_curve(Farx,Fary)[0],Normalize_curve(Farx,Fary)[1],'-')
plt.ylabel("Normalized bottom extension y")
plt.xlabel("Normalized bottom extension x")
plt.gca().invert_yaxis()
plt.title("Figure 4. Furthest point coordinates normalized")
plt.savefig("Figure_4_Furthest_point_coordinates_normalized.png")

(tck0), fp0, ier0, msg0 = splrep(Frame_count2, Furthest, s = 0, k = 3, full_output = True)
Framecountnew2 = np.arange(Frame_count2[0], max(Frame_count2), 10)
Pointvelsmooth = splev(Framecountnew, tck1, der = 1)
Pointvel = splev(Frame_count2, tck0, der = 1)
Pointvels = splev(Framecountnew2, tck1, der = 1)

plt.figure(5)
#plt.plot(Framecountnew,Pointvelsmooth,'-')
plt.plot(Frame_count2,Pointvel,'.')
plt.plot(Framecountnew,Pointvelsmooth,"-")
axes = plt.gca()
axes.set_ylim([-1000,1000])
plt.ylabel("Extension speed (microns/frame)")
plt.xlabel("Time (frames)")
plt.title("Figure 5. Velocity of the furthest point from the centroid")
plt.savefig("Figure_5_Velocity_of_the_furthest_point_of_the_proboscis_from_the_centroid.png")

plt.figure(6)
"""dy = np.zeros((np.asarray(Fary)).shape,np.float)
dy[0:-1] = np.diff(np.asarray(Fary))/np.diff(np.asarray(Frame_count2))
dy[-1] = (Fary[-1] - Fary[-2])/(Frame_count2[-1] - Frame_count2[-2])

dx = np.zeros((np.asarray(Farx)).shape,np.float)
dx[0:-1] = np.diff(Farx)/np.diff(Frame_count2)
dx[-1] = (Farx[-1] - Farx[-2])/(Frame_count2[-1] - Frame_count2[-2])"""
(tck4), fp4, ier4, msg4 = splrep(Frame_count2, Farx, s=0, k=3, full_output = True)
(tck5), fp5, ier5, msg5 = splrep(Frame_count2, Fary, s=0, k=3, full_output = True)

Botxvel = splev(znew, tck2, der=1)
Botyvel = splev(znew, tck3, der=1)
dx = splev(Frame_count2, tck4, der = 1)
dy = splev(Frame_count2, tck5, der = 1)
Approx_y = splev(Framecountnew2, tck3, der = 1)
Approx_x = splev(Framecountnew2, tck2, der = 1)

plt.plot(Frame_count2, dx, '.')
plt.plot(znew, Botxvel, '-')
plt.ylabel("X-Extension speed (microns/frame)")
plt.xlabel("Time (frames)")
plt.title("Figure 6. X-Velocity of the furthest point from the centroid")
axes = plt.gca()
axes.set_ylim([-30,30])
#==============================================================================
# plt.gca().invert_yaxis()
#==============================================================================
plt.savefig("Figure_6_X_Velocity_of_the_furthest_point.png")


plt.figure(7)
plt.plot(Frame_count2, dy , '.')
plt.plot(znew, Botyvel ,'-')
plt.ylabel("Y-Extension speed (microns/frame)")
plt.xlabel("Time (frames)")
axes = plt.gca()
axes.set_ylim([-30,30])
plt.title("Figure 7. Y-Velocity of the furthest point")
#==============================================================================
# plt.gca().invert_yaxis()
#==============================================================================
plt.savefig("Figure_7_Furthest_point_from_centroid_coordinates.png")

plt.figure(8)
plt.plot(Frame_count_mid,Mid_dist,'-')
plt.plot(Frame_count_mid,Mid_dist,'.')
plt.ylabel("Extension distance (microns)")
plt.xlabel("Time (frames)")
plt.title("Figure 8. Distance of the centroid of the proboscis mask")
plt.savefig("Figure_8_Distance_of_the_centroid_of_the_proboscis_mask.png")

plt.figure(9)
plt.plot(Ellipse_frames,Ellipse_mid_dist,'-')
plt.plot(Ellipse_frames,Ellipse_mid_dist,'.')
plt.ylabel("Extension distance (microns)")
plt.xlabel("Time (frames)")
plt.title("Figure 9. Distance of the centroid of the proboscis ellipse fit")
plt.savefig("Figure_9_Distance_of_the_centroid_of_the_proboscis_ellipse_fit.png")

plt.figure(10)
plt.plot(Centroid_x,Centroid_y,'.')
plt.plot(Centroid_x,Centroid_y,'-')
plt.ylabel("Centroid extension y")
plt.xlabel("Centroid extension x")
plt.title("Figure 10. Coordinates of the centroid motion")
plt.gca().invert_yaxis()
plt.savefig("Figure_10_Centroid_point_coordinates.png")

plt.show()

# Parameters for the total extension length and time taken.

Extension_length = max(Bot_dist)
Extension_length_centroid = max(Mid_dist)
Extension_length_ellipse = max(Ellipse_mid_dist)
Extension_length_furthest = max(Furthest) - Furthest[0]
Extension_duration = float(Frame_count2[Furthest.index(max(Furthest))] - Frame_count2[0])/163.317
Ext_dur_bot = float((Frame_count[Bot_dist.index(max(Bot_dist))])/163.317)

print "Total time for proboscis extension was: ", Extension_duration, "seconds", "\n"
print "Total extension distance was: ", Extension_length_furthest, "micrometers", "\n"
print "The weighted sum of the squared residuals of the spline approximations i.e. the tremor parameters are: ", "\n"
print "X-Velocity: ", fp2
print "Y-Velocity: ", fp3
print "Point-Velocity: ", fp1, "\n"
print "Furthest point coordinates: ",fp2+fp3 ,"\n"
print "(The further from 10 the value is the more tremor it represents)","\n"
print "The starting point is at distance: ", Furthest[0]
print "Bottom point extension distance: ", Extension_length, "micrometers", "\n"
print "Bottom point extension time: ", Ext_dur_bot
# Saving all the recorded data as text files of the normalized x and y values separated by space. This works with the 
# subsequent text reader function.

with open("Furthest_Point_Extension_Coordinates_Smoothened.txt", "wb") as f:
    csv.writer(f, delimiter = " ").writerows(np.vstack((Normalize_curve(Botxnew, Botynew))).T)
with open("Centroid_Extension_Coordinates.txt", "wb") as f:
    csv.writer(f, delimiter = " ").writerows(np.vstack((Normalize_curve(Centroid_x, Centroid_y))).T)
with open("Ellipse_Extension_Coordinates.txt", "wb") as f:
    csv.writer(f, delimiter = " ").writerows(np.vstack((Normalize_curve(Ellipse_mid_x, Ellipse_mid_y))).T)
with open("Furthest_Point_Extension_Coordinates_Unsmoothened.txt", "wb") as f:
    csv.writer(f, delimiter = ",").writerows((Normalize_curve(Farx, Fary)))
with open("Furthest_Point_Extension_Distance_Smoothened.txt", "wb") as f:
    csv.writer(f, delimiter = " ").writerows(np.vstack((Framecountnew,Botdistnew)).T)
with open("Velocity_of_Furthest_Point.txt", "wb") as f:
    csv.writer(f, delimiter = " ").writerows(np.vstack((Frame_count2,Pointvel)).T)
with open("X_Velocity_of_Furthest_Point.txt", "wb") as f:
    csv.writer(f, delimiter = " ").writerows(np.vstack((Frame_count2, dx)).T)
with open("Y_Velocity_of_Furthest_Point.txt", "wb") as f:
    csv.writer(f, delimiter = " ").writerows(np.vstack((Frame_count2, dy)).T)
    

# This function reads data from a file, plots it and finds the most optimal polynomial fit according
# to the fisher f test. The function requires the file name as the argument (or the whole location of
# the text file if it is not in the working directory)

def Best_Fit(File,residual):
    
# Loading the data from the text file and storing the columns of the file in arrays of x and y values.
    
    x = np.loadtxt(File,delimiter=" ",usecols=[0])
    y = np.loadtxt(File,delimiter=" ",usecols=[1])
    Filename = File[28:len(File)-4]
# Defining a function that will take a variable o as the order of the polynomial fit and returns the coefficients of
# the requested fit.
    
    def fit(o):
        fit = np.polyfit(x,y,o)
        p = np.polyval(fit,x)
        return p,fit
    
# Defining another function that calculates the residual of the polynomial fit. The function requires a fit as the
# variable.

    def Residual(fit):
        i = 0                               # Initial value in the array of x values.
        S = 0                               # Initial residual of 0.
        while i<len(x):                     # Loops over all the values in the array of x values.
            S = S+(fit[0][i] - y[i])**2          # Calculates the residual for each data point and adds them up.
            i = i+1
        return S
    
# Defining a function that calculates the fisher f values for all polynomial fits ranging from 0 to n where n is
# the variable that needs to be specified. The fisher f values are stored in an array in ascending order of the
# polynomial degree.
    
    def fisher(n):
        F = np.array([])
        for i in range(n):
            f = (((Residual(fit(i)) - Residual(fit(i+1)))/((i+1) - (i)))*((len(x)-(i+1))/Residual(fit(i+1))))
            F = np.append(F,f)
            i = i+1
        return F
    
# Specifying the critical f values for the given data that can be compared with the calculated f values later.
# Values taken from "http://www.itl.nist.gov/div898/handbook/eda/section3/eda3673.htm"
    
    Fcrit = [2.697, 2.465, 2.309, 2.196, 2.109, 2.040, 1.983, 1.936, 1.897, 1.863, 1.833, 1.807, 1.784, 1.764, 1.746, 1.729, 1.715, 1.702]
    
# Loop that increases the polynomial order after each iteration until the fisher f values of the given data are less
# than the critical values, in which case it stops and gives the optimal order of the polynomial. Starts at n = 0
# because the index of the first values in the arrays is 0.
    
    n = 2
    while (fisher(n+1)[n] > Fcrit[n] or Residual(fit(n)) > residual) and n < 10:
        n = n+1
    
    print "The fisher f values for first n+1 polynomials are: ", fisher(n+1),"\n"
    
    print "The critical f values for first n+1 polynomials are: ", Fcrit[0:4] , "\n"
    
    print "The optimal polynomial fit is of order ", n, "\n"
    
    print "The residual of the fit is ", Residual(fit(10))
    
# Plotting the data with the polynomial fit of the most optimal order superimposed on it.
    
    plt.plot(x,y, "-", x, fit(n)[0], "--")
    plt.ylabel("y-value")
    plt.xlabel("x-value")
    plt.title(Filename)
    #axes = plt.gca()
    #axes.set_ylim([min(fit(n)[0])-0.2*abs(min(fit(n)[0])),max(fit(n)[0])+0.2*abs(max(fit(n)[0]))])
    plt.savefig("Figure_11._" + Filename + ".png")
    plt.show()

# Returning the values of the most optimal polynomial fit.

    return n

#print Best_Fit(FileDir+"Velocity_of_Furthest_Point.txt",residual(Pointvel,Pointvels))
#print Best_Fit(FileDir+"Y_Velocity_of_Furthest_Point.txt", residual(dy,Approx_y))
#print Best_Fit(FileDir+"X_Velocity_of_Furthest_Point.txt", residual(dx,Approx_x))

"""
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def findDistance(r1,c1,r2,c2):
	d = (r1-r2)**2 + (c1-c2)**2
	d = d**0.5
	return d
		
#main function
cv2.namedWindow('tracker')

cap = cv2.VideoCapture(File)

while True:
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES,start+i)
        _,frame = cap.read()
        r = 1200./frame.shape[1]
        dim = (1200, int(frame.shape[0]*r))
        frame = cv2.resize(frame,dim,cv2.INTER_LINEAR)
        		#-----Drawing Stuff on the Image
        cv2.putText(frame,'Press a to start tracking',(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,color = (60,100,75),thickness = 3)
        		#cv2.rectangle(frame,(int(roi[0]),int(roi[1]),(int(roi[0]+roi[2]),int(roi[1]+roi[3]))),color = (100,255,100),thickness = 4)
                
        		#-----Finding ROI and extracting Corners
        frameGray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        Roi = frameGray[int(roi[1]):int(roi[1]+roi[3]),int(roi[0]):int(roi[0]+roi[2])] #selecting roi
                #roi = frame[int(roi[1]):int(roi[1]+roi[3]),int(roi[0]):int(roi[0]+roi[2])]
        new_corners = cv2.goodFeaturesToTrack(Roi,50,0.01,10) #find corners
        	
        		#-----converting to complete image coordinates (new_corners)
        	
        new_corners[:,0,0] = new_corners[:,0,0] + int(roi[0])
        new_corners[:,0,1] = new_corners[:,0,1] + int(roi[1])
        		 	 
        		#-----drawing the corners in the original image
        for corner in new_corners:
            cv2.circle(frame, (int(corner[0][0]),int(corner[0][1])) ,5,(0,255,0))
        	
        		#-----old_corners and oldFrame is updated
        oldFrameGray = frameGray.copy()
        old_corners = new_corners.copy()
        	
        cv2.imshow('tracker',frame)
        		
        a = cv2.waitKey(5)
        if a== 27:
            cv2.destroyAllWindows()
            cap.release()
        elif a == 97:
            break
        i = i +1
    i = 0
        	#----Actual Tracking-----
    while True:
        'Now we have oldFrame,we can get new_frame,we have old corners and we can get new corners and update accordingly'
	
		#read new frame and cvt to gray
        cap.set(cv2.CAP_PROP_POS_FRAMES,start+i)
        ret,frame = cap.read()
        r = 1200./frame.shape[1]
        dim = (1200, int(frame.shape[0]*r))
        frame = cv2.resize(frame,dim,cv2.INTER_LINEAR)
        frameGray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        		#finding the new tracked points
        new_corners, st, err = cv2.calcOpticalFlowPyrLK(oldFrameGray, frameGray, old_corners, None, **lk_params)
        	
        		#---pruning far away points:
        		#first finding centroid
        r_add,c_add = 0,0
        for corner in new_corners:
            r_add = r_add + corner[0][1]
            c_add = c_add + corner[0][0]
        centroid_row = int(1.0*r_add/len(new_corners))
        centroid_col = int(1.0*c_add/len(new_corners))
        		#draw centroid
        cv2.circle(frame,(int(centroid_col),int(centroid_row)),5,(255,0,0)) 
        		#add only those corners to new_corners_updated which are at a distance of 30 or lesse
        new_corners_updated = new_corners.copy()
        tobedel = []
        for index in range(len(new_corners)):
            if findDistance(new_corners[index][0][1],new_corners[index][0][0],int(centroid_row),int(centroid_col)) > 90:
                tobedel.append(index)
        new_corners_updated = np.delete(new_corners_updated,tobedel,0)
        	
        	
        
        		#drawing the new points
        for corner in new_corners_updated:
            cv2.circle(frame, (int(corner[0][0]),int(corner[0][1])) ,5,(0,255,0))
        if len(new_corners_updated) < 10:
            print 'OBJECT LOST, Reinitialize for tracking'
            break
        		#finding the min enclosing circle
        ctr , rad = cv2.minEnclosingCircle(new_corners_updated)
        	
        cv2.circle(frame, (int(ctr[0]),int(ctr[1])) ,int(rad),(0,0,255),thickness = 5)	
        		
        		#updating old_corners and oldFrameGray 
        oldFrameGray = frameGray.copy()
        old_corners = new_corners_updated.copy()
        	
        		#showing stuff on video
        cv2.putText(frame,'Tracking Integrity : Excellent %04.3f'%random.random(),(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,color = (200,50,75),thickness = 3)
        cv2.imshow('tracker',frame)
        	
        a = cv2.waitKey(5)
        if a== 27:
            cv2.destroyAllWindows()
            cap.release()
        elif a == 97:
            break	
        i = i+1

	
		
cv2.destroyAllWindows()	
"""


