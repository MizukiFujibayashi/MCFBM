# MCFBM
<p align="center"><img src="https://github.com/MizukiFujibayashi/MCFBM/blob/main/zebra1.png"></p>

Now published in : 



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
**fragment syllables**

Prepare over 2000 song files recorded in SAP2011, etc.

Execute a_make_data_plus_edge.m on MATLAB. Changing the threshold of Prewitt may improve syllable determination.


**Cluster the fragmented syllables**

Execute b_tsne_DBSCAN.m. Changing each parameter of DBSCAN may improve clustering.


**Generate a learning machine**

Execute c_AIC_CNN.m.

The entire process takes about 15 minutes in total.

**Translate song files into text (Optional)**

Execute d_corpas_plus_edge.m. Please align all parameters such as Prewitt's threshold with a_make_data_plus_edge.m.


**Translate real-time voicing**

Please execute syllable_detecter_edge.m on a computer connected to a microphone. At that time, please place the generated learning machine in the same directory.



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

This architecture was developed with python v3.8.8 using OpenCV library v4.7.0. 

The actual development environment is documented in the https://github.com/MizukiFujibayashi/MCFBM/requirements.txt.
