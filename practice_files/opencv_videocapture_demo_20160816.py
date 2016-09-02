# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 11:54:03 2016

@author: jack

NOTES
=====
This code is from this tutorial:
<https://tobilehman.com/blog/2013/01/20/extract-array-of-frames-from-mp4-using-python-opencv-bindings/>

There's also a stack exchange thread about it:
<http://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames>
"""

import cv2

vidcap = cv2.VideoCapture('bunny_720p_5mb.mp4')

success,image = vidcap.read()
# image is an array of array of [R,G,B] values

#%%
count = 0;
while success:
  success,image = vidcap.read()
  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
  if cv2.waitKey(10) == 27:                     # exit if Escape is hit
      break
  count += 1
  

count = 0;
while success:
  success,image = vidcap.read()
  if count == refFrameNum:
      refImg = image
  if cv2.waitKey(10) == 27:                     # exit if Escape is hit
      break
  count += 1