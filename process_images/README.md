image processing scripts
========================

These are python scripts to process images using various forms of background oriented schlieren.

### process_cgbos.py
This performs color gradient background oriented schlieren, which is similar to S-BOS but only takes the difference between the measured and reference image.  For more info, see [Color gradient background-oriented schlieren imaging](https://www.researchgate.net/publication/303324452_Color_gradient_background-oriented_schlieren_imaging)
### process_cgbos_with_extra_postprocessing.py
This performs CG-BOS but has additional postprocessing.
### process_dot_bos.py
Good old (well not that old) fashioned background oriented schlieren.  [OpenPIV](http://www.openpiv.net/openpiv-python/) is used to perform the cross correlation.
### process_sbos.py
This performs simplified BOS.  For more info see [Flow Visualization by a Simplified BOS Technique](https://www.researchgate.net/publication/268483453_Flow_Visualization_by_a_Simplified_BOS_Technique).
### process_sbos_with_extra_postprocessing.py
This performs simplified BOS with additional postprocessing.  This file requires the file coeff.txt in one of the steps, although to be honest it doesn't do much, You may be better off using process_sbos.py unless you're interested in playing around with some of the additional code.
