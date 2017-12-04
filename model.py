# 1. Import packages

import numpy as np
import pandas as pd
import sys
import os
import gensim
from tqdm import tqdm
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Activation
import logging

from utils.stemming import stemming_row
from utils.tokens import word_tokens
from utils.dictionary import create_dict
from utils.tfidf import tfidf
from utils.longest import longest_question
from utils.distance import distance

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

# 2. User setting 1: Import data
# Important: data needs to be stored in directory 'data' in parent folder of current working directory
path = os.getcwd()
os.chdir(path)
train_df = pd.read_csv("data/train_data.csv", delimiter=',')
test_df = pd.read_csv("data/test_data.csv", delimiter=',')
train_duplicate = pd.read_csv("data/train_labels.csv", delimiter=',')

#questions = list(train_df['question1'].values.astype('U')) + list(train_df['question2'].values.astype('U'))


# -------------------------------------------- SECTION 2 
# As you see, we start over with `train_df` in TFIDF

# 7. TF-IDF 
train_df['q1_feats'] = tfidf(train_df['question1'])
train_df['q2_feats'] = tfidf(train_df['question2'])
test_df['q1_feats'] = tfidf(test_df['question1'])
test_df['q2_feats'] = tfidf(test_df['question2'])


# 8. Train set
train_longest = longest_question(train_df['q1_feats'], train_df['q2_feats'])
test_longest = longest_question(test_df['q1_feats'], test_df['q2_feats'])

train_questions1 = sequence.pad_sequences(train_df['q1_feats'], train_longest)
train_questions2 = sequence.pad_sequences(train_df['q2_feats'], train_longest)

test_questions1 = sequence.pad_sequences(test_df['q1_feats'], test_longest)
test_questions2 = sequence.pad_sequences(test_df['q2_feats'], test_longest)

train_distances = distance(train_questions1, train_questions2)
test_distances = distance(test_questions1, test_questions2)


model = Sequential()
model.add(Dense(64, input_dim=1, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

#print('distances: ', distances)

model.fit(train_distances, train_duplicate['is_duplicate'], epochs=10, batch_size=32)
predictions = model.predict(test_distances, verbose=1)

rounded = [int(round(x[0])) for x in predictions]

submission_df = pd.DataFrame(index=test_df.test_id, columns=['is_duplicate'], dtype=np.uint)
submission_df.index.name = 'test_id'
submission_df.is_duplicate = rounded

submission_df.to_csv('output/submission.csv')

print('predications: ', predictions)

#for counter, distance in enumerate(distances): 
	#print(counter, '  distance:', distance, ' is duplicate: ', train_duplicate['is_duplicate'][counter])

