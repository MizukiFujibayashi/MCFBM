# MCFBM
<p align="center"><img src="https://github.com/MizukiFujibayashi/MCFBM/blob/main/zebra1.png"></p>

1. [About](#About)
2. [Installation](#[Installation])
3. [Required](#[Required])



## About

These softwares are for analysing finch vocalizations.



## Installation

Installation will be completed instantly.

**Clone this repository**

```bash
$ git clone https://github.com/MizukiFujibayashi/MCFBM.git
```


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

###Developed

**python v3.8.8**

cv2 4.7.0

numpy 1.21.4

math

matplotlib.pyplot

os

pandas 1.4.2

seaborn

shelve
