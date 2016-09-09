animation files visualizing the S-BOS algorithm
===============================================

This directory contains several python scripts that output animations in mp4 format.  These animations were made for illustrating the fact that for S-BOS, each location along the waveform of the background pattern has a unique and nonlinear relationship between the outputted intensity, and the displacement of the image at that point.

Because these loop through each displacement at each location on the background pattern waveform, the file create_animation_brightness_vs_shift_along_waveform.py was used to output an array of polynomial coefficients that were used in some processing scripts.  This approach to correcting for the nonlinearity of the output intensity was later set aside in favor of correcting for the nonlinearity by dividing the difference of the measured and reference images by the average gradient of the measured and reference images.
