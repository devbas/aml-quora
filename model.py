# 1. Import packages

import numpy as np
import pandas as pd
import os
import gensim
from tqdm import tqdm

import utils.stemming as stemming
import utils.tokens as tokens
import utils.dictionary as dictionary


# 2. User setting 1: Import data
# Important: data needs to be stored in directory 'data' in parent folder of current working directory
path = os.getcwd()
os.chdir(path)
train_df = pd.read_csv("data/train_data.csv", nrows=1000, delimiter=',')

questions = list(train_df['question1']) + list(train_df['question2'])

# 3. Split text
print('Split text:')
c = 0
for question in tqdm(questions): 
	questions[c] = tokens.word_tokens(question)


# 4. Stemming 
print('Stemming:')
c = 0
for question in tqdm(questions): 
	questions[c] = stemming.stemming_row(question)

# 5. Train model 
model = gensim.models.Word2Vec(questions, size=300, workers=16, iter=10, negative=20)

# trim memory
model.init_sims(replace=True)

# create a dict 
w2v = dict(zip(model.wv.index2word, model.wv.syn0))
print("Number of tokens in Word2Vec:", len(w2v.keys()))


# 6. Save model
model.save('output/word2vec.mdl')
model.wv.save_word2vec_format('output/word2vec.bin', binary=True)


# 5. Word to dictionary
# train_index = train_stemmed.apply(dictionary.create_dict, axis=1, raw=True)