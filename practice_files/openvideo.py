# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 13:52:47 2016

@author: jack

NOTES
=====
At first I couldn't get it to even load a video.  It could connect to the 
webcam but not to a video file.  I found a workaround- someone on stackexchange
said that it would play if you converted it with certain setting using 
mencoder.  Now it appears to play.  I just need to be able to convert it to 
an array I can work with.  So far I have not been able to do this.

"frame" is a mat object.  I can convert it to an array using np.array.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

#cam = cv2.VideoCapture('street.avi')
cam = cv2.VideoCapture('/home/jack/Pictures/2016-08-05_flame/DSC_0023.MOV')
print cam.isOpened()

width = cam.get(4)
height = cam.get(3)
if cam.get(0) > 0:
    nFrames = cam.get(0)
else:
    nFrames = 10
vidData = [width, height, nFrames]

vid = np.zeros((width,height,3,nFrames))
frameNum = 0
while(True):
        ret, frame = cam.read()
        
        if frameNum == 0:
            cv2.imshow('frame', frame)
            
        a = np.array(frame[:,:,:])
        a2 = np.expand_dims(a, 4)
        vid[:,:,:,frameNum] = a2[:,:,:,0]
        
        frameNum = frameNum + 1
        print frameNum
        print ret
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()


#%%

# create videocapture object
filename = '/home/jack/Pictures/2016-08-05_flame/DSC_0023.MOV'
cam = cv2.VideoCapture(filename)

# check if the videocapture object was successfully created
if cam.isOpened() == 'False':
    print('error reading ' + str(filename))

# get video dimensions
width = cam.get(4)
height = cam.get(3)
if cam.get(0) > 0:
    nFrames = cam.get(0)
else:
    nFrames = 10
vidData = [width, height, nFrames]
# NOTE: RIGHT NOW THE METHOD I USE TO CONVERT VIDEO DOESNT KEEP METADATA ON NFRAMES I THINK
vid = np.zeros((width,height,3,nFrames))


for frameNum in range(nFrames):
    ret, frame = cam.read()
    
    a = np.array(frame[:,:,:])
    a2 = np.expand_dims(a, 4)
    vid[:,:,:,frameNum] = a2[:,:,:,0]

    
    print frameNum
    print ret
    
    #cv2.imshow('frame', frame)
    #if cv2.waitKey(10) & 0xFF == ord('q'):
    #    break
    #cam.release()
    #cv2.destroyAllWindows()

# announce that the video was read successfully
print('successfully read 20 frames')
#plt.figure(), plt.imshow(Vid[])





