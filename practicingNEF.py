# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 18:56:21 2016
@author: jack
I'm learning how to use NEF (Nikon Electronic Format) to separate color channels
"""

import rawpy
import imageio
import matplotlib.pyplot as plt
import numpy as np

Tk().withdraw()
file_opt = options = {}
options['filetypes'] = [('all files', '.*')]
options['initialdir'] = '/home/jack/Pictures'
filename = askopenfilename(**file_opt) 

if filename[-4:] =='.JPG':
    rgb = plt.imread(filename)
else:
    raw = rawpy.imread(filename)
    rgb = raw.postprocess()


#%% Three subplots sharing both x/y axes
f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
ax1.imshow(rgb[:,:,0],cmap='gray')
ax2.imshow(rgb[:,:,1],cmap='gray')
ax3.imshow(rgb[:,:,2],cmap='gray')

f.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)




fig2=plt.figure()
plt.plot(rgb[1500,:,0],'r')
plt.plot(rgb[1500,:,1],'g')
plt.plot(rgb[1500,:,2],'b')

#%% create image
X = np.linspace(1,1000,1000)
R = np.cos(X*2*np.pi/1000)/2+0.5
G = np.cos(X*2*np.pi/1000+2*np.pi/3)/2+0.5
B = np.cos(X*2*np.pi/1000+4*np.pi/3)/2+0.5
plt.figure()
plt.plot(R)
plt.plot(G)
plt.plot(B)

R2 = np.resize(R,(1000,800))
G2 = np.resize(G,(1000,800))
B2 = np.resize(B,(1000,800))
plt.figure()
plt.imshow(R2)


RGB = np.zeros((1000,800,3))
RGB[:,:,0] = R2
RGB[:,:,1] = G2
RGB[:,:,2] = B2

plt.figure()
plt.imshow(RGB)

