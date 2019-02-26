# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 10:58:45 2019

@author: nigo0024
"""
import os
from PIL import Image

def matrix_plots(base_dir, plots_wide=5, plots_high=4):
    fnames_in = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))]
    fnames_in.sort()

    img = Image.open(fnames_in[0])
    width, height = img.size
    width_new = width * plots_wide
    height_new = height * plots_high

    img_out = Image.new('RGB', (width_new, height_new), (255, 255, 255))
    row_n = -1
    col_n = 0
    for idx, fname in enumerate(fnames_in):
        img = Image.open(fname)
        width, height = img.size
        col_n = idx % plots_wide
        if col_n == 0:
            row_n += 1
    #    if row_n > 2:
    #        break
        img_out.paste(img, box=(col_n * width, row_n * height))
        img = None
    base_dir_out = os.path.join(os.path.dirname(base_dir), 'matrix_plots')
    if not os.path.isdir(base_dir_out):
        os.mkdir(base_dir_out)
    fname_out = os.path.join(base_dir_out, os.path.basename(base_dir) + '.png')
    img_out.save(fname_out)
    img_out = None

# In[trad_0050]
base_dir = r'G:\SOIL\GIS\SNS\eonr\2019-02-19\imperial\trad_0050'
matrix_plots(base_dir)
# In[trad_0075]
base_dir = os.path.join(os.path.dirname(base_dir), 'trad_0075')
matrix_plots(base_dir)
# In[trad_0100]
base_dir = os.path.join(os.path.dirname(base_dir), 'trad_0100')
matrix_plots(base_dir)
# In[trad_0125]
base_dir = os.path.join(os.path.dirname(base_dir), 'trad_0125')
matrix_plots(base_dir)
# In[trad_0150]
base_dir = os.path.join(os.path.dirname(base_dir), 'trad_0150')
matrix_plots(base_dir)
# In[trad_0175]
base_dir = os.path.join(os.path.dirname(base_dir), 'trad_0175')
matrix_plots(base_dir)
# In[trad_0200]
base_dir = os.path.join(os.path.dirname(base_dir), 'trad_0200')
matrix_plots(base_dir)
# In[trad_0250]
base_dir = os.path.join(os.path.dirname(base_dir), 'trad_0250')
matrix_plots(base_dir)

# In[social_0103_0010]
base_dir = r'G:\SOIL\GIS\SNS\eonr\2019-02-19\imperial\social_0103_0010'
matrix_plots(base_dir)
# In[social_0125_0100]
base_dir = os.path.join(os.path.dirname(base_dir), 'social_0125_0100')
matrix_plots(base_dir)
# In[social_0163_0250]
base_dir = os.path.join(os.path.dirname(base_dir), 'social_0163_0250')
matrix_plots(base_dir)
# In[social_0225_0500]
base_dir = os.path.join(os.path.dirname(base_dir), 'social_0225_0500')
matrix_plots(base_dir)
# In[social_0350_1000]
base_dir = os.path.join(os.path.dirname(base_dir), 'social_0350_1000')
matrix_plots(base_dir)
# In[social_0600_2000]
base_dir = os.path.join(os.path.dirname(base_dir), 'social_0600_2000')
matrix_plots(base_dir)
# In[social_0850_3000]
base_dir = os.path.join(os.path.dirname(base_dir), 'social_0850_3000')
matrix_plots(base_dir)
# In[social_1350_5000]
base_dir = os.path.join(os.path.dirname(base_dir), 'social_1350_5000')
matrix_plots(base_dir)

# In[trad_0050]
base_dir = r'G:\SOIL\GIS\SNS\eonr\2019-02-19\metric\trad_28'
matrix_plots(base_dir)
# In[trad_0075]
base_dir = os.path.join(os.path.dirname(base_dir), 'trad_42')
matrix_plots(base_dir)
# In[trad_0100]
base_dir = os.path.join(os.path.dirname(base_dir), 'trad_56')
matrix_plots(base_dir)
# In[trad_0125]
base_dir = os.path.join(os.path.dirname(base_dir), 'trad_70')
matrix_plots(base_dir)
# In[trad_0150]
base_dir = os.path.join(os.path.dirname(base_dir), 'trad_84')
matrix_plots(base_dir)
# In[trad_0175]
base_dir = os.path.join(os.path.dirname(base_dir), 'trad_98')
matrix_plots(base_dir)
# In[trad_0200]
base_dir = os.path.join(os.path.dirname(base_dir), 'trad_112')
matrix_plots(base_dir)
# In[trad_0250]
base_dir = os.path.join(os.path.dirname(base_dir), 'trad_140')
matrix_plots(base_dir)

# In[social_0103_0010]
base_dir = r'G:\SOIL\GIS\SNS\eonr\2019-02-19\metric\social_57_0022'
matrix_plots(base_dir)
# In[social_0125_0100]
base_dir = os.path.join(os.path.dirname(base_dir), 'social_70_0220')
matrix_plots(base_dir)
# In[social_0163_0250]
base_dir = os.path.join(os.path.dirname(base_dir), 'social_91_0551')
matrix_plots(base_dir)
# In[social_0225_0500]
base_dir = os.path.join(os.path.dirname(base_dir), 'social_126_1102')
matrix_plots(base_dir)
# In[social_0350_1000]
base_dir = os.path.join(os.path.dirname(base_dir), 'social_196_2205')
matrix_plots(base_dir)
# In[social_0600_2000]
base_dir = os.path.join(os.path.dirname(base_dir), 'social_336_4409')
matrix_plots(base_dir)
# In[social_0850_3000]
base_dir = os.path.join(os.path.dirname(base_dir), 'social_476_6614')
matrix_plots(base_dir)
# In[social_1350_5000]
base_dir = os.path.join(os.path.dirname(base_dir), 'social_756_11023')
matrix_plots(base_dir)