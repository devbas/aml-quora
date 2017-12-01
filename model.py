# 1. Import packages

import numpy as np
import pandas as pd
import os

import utils.stemming as stemming
import utils.tokens as tokens
import utils.dictionary as dictionary


# 2. User setting 1: Import data
# Important: data needs to be stored in directory 'data' in parent folder of current working directory
path = os.getcwd()
os.chdir(path)
train_df = pd.read_csv("data/train_data.csv", nrows=10, delimiter=',')


# 3. Split text
train_tokenized = train_df.apply(tokens.word_tokens, axis=1, raw=True)


# 4. Stemming 
train_stemmed = train_tokenized.apply(stemming.stemming_row, axis=1, raw=True)


# 5. Word to dictionary
train_index = train_stemmed.apply(dictionary.create_dict, axis=1, raw=True)