# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 19:04:50 2016
@author: jack
This script is for generating color background patterns for bidirectional S-BOS
"""

import generateBackgroundImage as bg
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import cv2


def generateBackgroundImage(width,height,N,waveform,orientation):
    import numpy as np
    import cv2
    from scipy.signal import kaiserord, lfilter, firwin, freqz, square
    from scipy import signal
    
    if orientation == 'vertical' or orientation == 'v' or orientation == 'V':
        W=width
        width = height
        height = W
    x = np.linspace(0, N*2*np.pi, height)
    if waveform == 'square' or waveform == 'sq' or waveform == 'SQ':
        y = signal.square(x)
    else:
        y = np.sin(x)
    
    Y = np.expand_dims(y, 1)
    #Y =np.resize(Y, (height,width)
    
    while Y.size < height*width:
        Y = np.append(Y, Y, axis=1)
        
    Y2 = Y[1:height, 1:width]
    
    if orientation == 'vertical' or orientation == 'v' or orientation == 'V':
        Y2=np.rot90(Y2, k=1)
    
    return Y2


#%% generate background that will be assigned to first color channel
width=1920                                                                      # select width
height=1080                                                                     # select height
wavelength1 = 6                                                                 # select wavelength
waveform1='sq'                                                                   # select waveform
orientation1='V'                                                                # select orientation
if orientation1 == 'H':
    N1 = height/wavelength1
else:
    N1 = width/wavelength1
q1 = bg.generateBackgroundImage(width,height,N1,waveform1,orientation1)


#%% generate background that will be assigned to second color channel
wavelength2 = 6                                                                # select wavelength
waveform2='sq'                                                                   # select waveform
orientation2='H'                                                                # select orientation
if orientation2 == 'H':
    N2 = height/wavelength2
else:
    N2 = width/wavelength2
q2 = bg.generateBackgroundImage(width,height,N2,waveform2,orientation2)


#%% assemble the two backgrounds to one RGB image
img = np.zeros((height-1,width-1,3))
img[:,:,0]=q1/2+0.5
img[:,:,2]=q2/2+0.5


#%% disply the background
fig1=plt.figure()
plt.imshow(img)
plt.draw()
plt.show()
fig2=plt.figure()
plt.close(fig2)


#%% prompt user in the console to choose whether to save
filename = 'BG_' + str(waveform1) + '_' + str(waveform2) + '.jpg'
print('suggested filename: ' + filename)
print('press enter to accept, or type a new name to change it.  press space then enter to skip.')
userInput = raw_input()
if len(userInput) == 0:
    scipy.misc.imsave(filename, img)
    print('file saved as ' + filename)
elif len(userInput) == 1:
    print('you have chosen not to save the file')
elif len(userInput) > 1:
    print('input desired filename.  be sure to include a file extention')
    scipy.misc.imsave(userInput, img)
    print('file saved as ' + userInput)