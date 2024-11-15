# MCFBM
<p align="center"><img src="https://github.com/MizukiFujibayashi/MCFBM/blob/main/zebra1.png"></p>

Now published in : https://doi.org/10.1016/j.crmeth.2024.100844



## About

This software is for analyzing songbird behavior within an efficient and cost-effective marker-based motion capture system even applicable to multiple subjects tracking.

Further, it could be applied to track any distinctly coloured objects.


## Installation

Installation will be completed instantly.

**Clone this repository**

```bash
$ git clone https://github.com/MizukiFujibayashi/MCFBM.git
```

## Usage

Functions for tracking markers and bird location and body direction estimation are in https://github.com/MizukiFujibayashi/mocap_functions.py

Several functions to summarize data for behavioral analysis are included.

### Tracking markers

**For single subject tracking** : `mocap_functions.track`  
This also provides the x,y-coordinates of the center of bird shillhouete aquired by MoG background subtraction.

**For multiple subjects tracking** : `mocap_functions.track_mult`  
This is applicable to single subject tracking. 

You need to specify several parameters including the initial position of markers. All parameters do not have to be specified very strictly as the combination of parameters would contribute to filter noise.

### Body location and direction estimation

**For single subject** : `mocap_functions.approx_body_ellipse`  

**For multiple subjects** : `mocap_functions.approx_body_ellipse_mult`  
This is also applicable for single subject. 

## Required

**python**

cv2

numpy

math

matplotlib.pyplot

os

pandas

seaborn

shelve

### Developed

This architecture was developed with python v3.8.8 using OpenCV library v4.7.0.72 
