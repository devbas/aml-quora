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
from nltk.stem.porter import *
#from keras.preprocessing.text import one_hot
from gensim.models import word2vec
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

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
'''stop_words = set(stopwords.words("english"))

def stopword_removal(row): 

  output = row;
  output['question1'] = [w for w in row['question1'] if not w in stop_words]
  output['question2'] = [w for w in row['question2'] if not w in stop_words]

  return output

train_tokenized = train_tokenized.apply(stopword_removal, axis=1, raw=True)'''


# 5. Stemming 
stemmer = PorterStemmer()

def stemming_row(row): 

  output = row; 
  output['question1'] = [stemmer.stem(w) for w in row['question1']]
  output['question2'] = [stemmer.stem(w) for w in row['question2']]

  return output

train_tokenized = train_tokenized.apply(stemming_row, axis=1, raw=True)


# 6. Set vocabulary/Dictionary 
sentences = train_tokenized['question1'];

num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

print('sentences: ', sentences);

model = word2vec.Word2Vec(sentences, size=num_features, window = context)

model_name = "300features_40minwords_10context"
model.save(model_name)

'''def calculate_vocabulary_size(row): 
  row_size = len(row['question1']) + len(row['question2']) 

  return row_size

vocabulary_size = train_tokenized.apply(calculate_vocabulary_size, axis=1, raw=True)'''

#print('question1: ', len(train_tokenized['question1']))

#def create_library(row): 


# 7. TF IDF
