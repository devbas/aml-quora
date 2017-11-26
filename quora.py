''' Prerequisites: The project uses Python3 among the following libraries: 

'''

# 1. Import packages

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from numpy import genfromtxt

# 2. User setting 1: Import data
# Important: data needs to be stored in directory 'data' in parent folder of current working directory

path = os.getcwd()
os.chdir(path)
raw_df = pd.read_csv("data/test_data.csv")
print('number of observations: {:,}'.format(raw_df.shape[0]))