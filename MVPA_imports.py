#!/usr/bin/env python
# coding: utf-8

# In[1]:


# imports

import os
## --- ##

import numpy as np
import pandas as pd

from glob import glob

import nibabel as nib
from nilearn import image
from nilearn import masking
from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn import plotting

import itertools

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score
from sklearn import linear_model


from scipy.stats import ttest_rel
from scipy.stats import pearsonr
from matplotlib import pyplot as plt 

# import function that corrects pattern drift
