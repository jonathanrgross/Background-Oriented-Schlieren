# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 15:43:08 2016
@author: jack
This is for troubleshooting my problems with being unable to find the 
wavelength and phase shift
"""

import numpy as np
from matplotlib import pyplot as plt
pi = np.pi


def refine_wavelength(v1, wavelength):
    x = np.array(range(int(wavelength*100)))
    count = 0
    shift = np.linspace(-wavelength/4, wavelength/4, 5000)
    maxcor = np.zeros(len(shift))
    
    for i in shift:
        WL = wavelength+i
        v2 = np.sin(x*(2*pi)/WL)
    
        cor = np.correlate(v1, v2, mode='full')
    
        maxcor[count] = np.max(cor)
        count = count + 1
    
    plt.figure()
    plt.plot(shift,maxcor)
    newwavelength = wavelength + shift[np.argmax(maxcor)]
    print('the wavelength is ' + str(newwavelength))
    return(newwavelength)



def find_phase(v1, wavelength):
    x = np.array(range(int(6*wavelength)))
    y = np.sin((x)*(2*pi)/wavelength)
    
    cor = np.correlate(y[0:int(2*wavelength)], v1, mode='valid')
    
    phaseoffset = wavelength - (len(cor) - np.argmax(cor*range(len(cor))) - 1)
    print('the phase shift is ' + str(phaseoffset))
    return(phaseoffset)



#%% test the function to refine the wavelength
# create data
wavelength1 = 7.5632
wavelength2 = 9.3
ps1 = 3.6
ps2 = 2.6
x = np.array(range(1000))
yr = 2.1*np.sin((x+ps1)*(2*pi)/wavelength1)


# call refine_wavelength
newwavelength = refine_wavelength(yr, wavelength1)

# output result
print('the wavelength is ' + str(newwavelength))



#%% figuring out how to make the code to find the phase

newwavelength = 23.1
ps1 = 0
ps2 = 5.66
x = np.array(range(int(6*wavelength)))
yr = np.sin((x+ps1)*(2*pi)/newwavelength)
y = np.sin((x+ps2)*(2*pi)/newwavelength)

plt.figure()
plt.plot(x,yr)
plt.plot(x,y)
plt.title('waveforms being compared')

cor = np.correlate(y[0:int(2*wavelength)], yr, mode='valid')

plt.figure()
plt.plot(cor)
plt.title('cross correlation output')

#phaseoffset = np.argmax(cor)%wavelength1
phaseoffset = len(cor) - np.argmax(cor*range(len(cor))) - 1

print('the phase shift is ' + str(phaseoffset))


#%% test the find_phase function
wavelength = 31.11
ps = 30
x = np.array(range(300))
v1 = np.sin((x+ps)*(2*pi)/wavelength)

phase = find_phase(v1, wavelength)

#%% I don't know what this is

plt.figure()
plt.plot(x[:60],yr[:60])
for i in range(10):
    y = np.sin((x+i)*(2*pi)/wavelength1)
    plt.plot(x[:60],y[:60])






