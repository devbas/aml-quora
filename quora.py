''' Prerequisites: 

- The project uses Python3
- Make sure you have the nltk corpus installed:
	* >>> import nltk 
	* >>> nltk.download()
	* Download all collections 

'''

# 1. Import packages

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# 2. User setting 1: Import data
# Important: data needs to be stored in directory 'data' in parent folder of current working directory

path = os.getcwd()
os.chdir(path)
train_df = pd.read_csv("data/train_data.csv", nrows=20, delimiter=',')
#train_df.head()
print('number of observations: {:,}'.format(train_df.shape[0]))


# 3. Split text
def word_tokens(row): 

	output = row;
	output['question1'] = word_tokenize(str(row['question1']).lower())
	output['question2'] = word_tokenize(str(row['question2']).lower())

	return output

train_tokenized = train_df.apply(word_tokens, axis=1, raw=True)


# 4. Stopword removal
stop_words = set(stopwords.words("english"))

def stopword_removal(row): 

	output = row;
	output['question1'] = [w for w in row['question1'] if not w in stop_words]
	output['question2'] = [w for w in row['question2'] if not w in stop_words]

	return output

train_tokenized = train_tokenized.apply(stopword_removal, axis=1, raw=True)

print('train tokenized: ', train_tokenized);