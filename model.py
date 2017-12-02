# 1. Import packages

import numpy as np
import pandas as pd
import os
import gensim
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.stemming import stemming_row
from utils.tokens import word_tokens
from utils.dictionary import create_dict
from utils.vec_features import vec_features


# 2. User setting 1: Import data
# Important: data needs to be stored in directory 'data' in parent folder of current working directory
path = os.getcwd()
os.chdir(path)
train_df = pd.read_csv("data/train_data.csv", delimiter=',')

questions = list(train_df['question1']) + list(train_df['question2'])

# 3. Split text
print('Split text:')
c = 0
for question in tqdm(questions): 
  questions[c] = word_tokens(question)

print('questions: ', questions)

# 4. Stemming 
print('Stemming:')
c = 0
for question in tqdm(questions): 
  questions[c] = stemming_row(question)

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


# 7. TF-IDF 
tfidf = TfidfVectorizer(lowercase=False)
tfidf.fit_transform(questions)

word2tfidf = dict(zip(tfidf.get_feature_names(), tfidf.idf_))

train_df['q1_feats'] = vec_features(train_df['question1'], word2tfidf)
train_df['q2_feats'] = vec_features(train_df['question2'], word2tfidf)



# 8. Validate model 
train_df = train_df.reindex(np.random.permutation(train_df.index))

# set number of train and test instances
num_train = int(train_df.shape[0] * 0.88)
num_test = train_df.shape[0] - num_train 

print("Number of training pairs: %i"%(num_train))
print("Number of testing pairs: %i"%(num_test))

# init data data arrays
X_train = np.zeros([num_train, 2, 300])
X_test  = np.zeros([num_test, 2, 300])
Y_train = np.zeros([num_train]) 
Y_test = np.zeros([num_test])