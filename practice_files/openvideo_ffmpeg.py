# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 15:10:48 2016

@author: jack
"""

import subprocess as sp
command = [ FFMPEG_BIN,
            '-i', '20160529.avi',
            '-f', 'image2pipe',
            '-pix_fmt', 'rgb24',
            '-vcodec', 'rawvideo', '-']
pipe = sp.Popen(command, stdout = sp.PIPE, bufsize=10**8)