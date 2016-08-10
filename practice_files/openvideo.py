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

cam = cv2.VideoCapture('street.avi')
print cam.isOpened()

vid = np.zeros((600,360,3,80))
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



