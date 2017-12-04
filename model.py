# 1. Import packages

import numpy as np
import pandas as pd
import sys
import os
import gensim
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Activation
import logging

from utils.stemming import stemming_row
from utils.tokens import word_tokens
from utils.dictionary import create_dict
from utils.vec_features import vec_features
from utils.euclidean_distance import euclidean_distance
from utils.longest import longest_question

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

# 2. User setting 1: Import data
# Important: data needs to be stored in directory 'data' in parent folder of current working directory
path = os.getcwd()
os.chdir(path)
train_df = pd.read_csv("data/train_data.csv", delimiter=',')
train_duplicate = pd.read_csv("data/train_labels.csv", delimiter=',')

questions = list(train_df['question1'].values.astype('U')) + list(train_df['question2'].values.astype('U'))


# -------------------------------------------- SECTION 2 
# As you see, we start over with `train_df` in TFIDF

# 7. TF-IDF 
tfidf = TfidfVectorizer(lowercase=False)
tfidf.fit_transform(questions)

word2tfidf = dict(zip(tfidf.get_feature_names(), tfidf.idf_)) # To research

print('Features to vec')
train_df['q1_feats'] = vec_features(train_df['question1'], word2tfidf)
train_df['q2_feats'] = vec_features(train_df['question2'], word2tfidf)


# 8. Train set
longest = longest_question(train_df['q1_feats'], train_df['q2_feats'])

questions1 = sequence.pad_sequences(train_df['q1_feats'], longest)
questions2 = sequence.pad_sequences(train_df['q2_feats'], longest)


distances= np.zeros(len(questions1))
counter = 0

for q1, q2 in zip(questions1, questions2): 
	distance = euclidean_distance(q2, q1)
	distances[counter] = distance
	counter += 1

model = Sequential()
model.add(Dense(64, input_dim=1, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

print('distances: ', distances)

model.fit(distances, train_duplicate['is_duplicate'], epochs=10, batch_size=32)

#for counter, distance in enumerate(distances): 
	#print(counter, '  distance:', distance, ' is duplicate: ', train_duplicate['is_duplicate'][counter])

