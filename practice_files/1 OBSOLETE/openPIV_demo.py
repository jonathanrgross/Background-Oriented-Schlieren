# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 14:17:33 2016
@author: jack
This is a tutorial for openpiv.  I downloaded it from here:
http://www.openpiv.net/openpiv-python/src/tutorial.html

Aug 4- now that I have this implemented in the actual code I think this is 
obsolete and can probably be deleted.
"""

import openpiv.tools
import openpiv.process
import openpiv.scaling
import numpy as np
import matplotlib.pyplot as plt
import cv2

# read images in.  add something to let the user select the images from a dialog
frame_a  = openpiv.tools.imread( '/home/jack/Pictures/dots1.jpg' )
frame_b  = openpiv.tools.imread( '/home/jack/Pictures/dots1rot.jpg' )

# use logical indexing to set upper limit on brightness
frame_b[frame_b > 200] = 0

#%% use contrast limited adaptive histogram equalization (CLAHE)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(frame_a)

# row and column sharing
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
ax1.imshow(frame_a, cmap='gray')
ax2.imshow(cl1, cmap='gray')
ax3.hist(frame_b.ravel(), bins=256, fc='k', ec='k')
ax4.hist(cl1.ravel(), bins=256, fc='k', ec='k')


#%% perform PIV
A=frame_a.astype(np.int32)
B=frame_b.astype(np.int32)
u, v, sig2noise = openpiv.process.extended_search_area_piv( A.astype(np.int32), B.astype(np.int32), window_size=24, overlap=12, dt=0.02, search_area_size=64, sig2noise_method='peak2peak' )

x, y = openpiv.process.get_coordinates( image_size=frame_a.shape, window_size=24, overlap=12 )

# plot the vector field
plt.figure()
plt.quiver( x, y, u, v )

#%%
# invalid vectors are set to NaN
u, v, mask = openpiv.validation.sig2noise_val( u, v, sig2noise, threshold = 1.3 )

# this interpolates the missing vectors
u, v = openpiv.filters.replace_outliers( u, v, method='localmean', kernel_size=2)

# scale the results so they're in correct units
x, y, u, v = openpiv.scaling.uniform(x, y, u, v, scaling_factor = 96.52 )

# plot the vector field
plt.figure()
plt.quiver( x, y, u, v )

# save the output vectors
#openpiv.tools.save(x, y, u, v, filename='dots_output_20160712.txt' , fmt='%6.3f', delimiter='       ')
np.savetxt("dots_output_20160712_u.csv", u, delimiter=",")
np.savetxt("dots_output_20160712_v.csv", v, delimiter=",")
np.savetxt("dots_output_20160712_x.csv", x, delimiter=",")
np.savetxt("dots_output_20160712_y.csv", y, delimiter=",")
