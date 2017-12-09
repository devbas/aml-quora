import numpy as np
import pandas as pd
import sys
import os
from nltk.stem.porter import *
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
import xgboost as xgb
from sklearn.metrics import log_loss

from utils.tfidf import tfidf
from utils.longest import longest_question
from utils.distance import distance

# 2. User setting 1: Import data
# Important: data needs to be stored in directory 'data' in parent folder of current working directory
path = os.getcwd()
os.chdir(path)
train_df = pd.read_csv("data/train_data.csv", delimiter=',')
test_df = pd.read_csv("data/test_data.csv", delimiter=',')
train_duplicate = pd.read_csv("data/train_labels.csv", delimiter=',')

stemmer = PorterStemmer()

def filter(text): 

	text = str(text)
	text = text.lower()

    # Clean the text
	text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
	text = re.sub(r"what's", "what is ", text)
	text = re.sub(r"\'s", " ", text)
	text = re.sub(r"\'ve", " have ", text)
	text = re.sub(r"can't", "cannot ", text)
	text = re.sub(r"n't", " not ", text)
	text = re.sub(r"i'm", "i am ", text)
	text = re.sub(r"\'re", " are ", text)
	text = re.sub(r"\'d", " would ", text)
	text = re.sub(r"\'ll", " will ", text)
	text = re.sub(r",", " ", text)
	text = re.sub(r"\.", " ", text)
	text = re.sub(r"!", " ! ", text)
	text = re.sub(r"\/", " ", text)
	text = re.sub(r"\^", " ^ ", text)
	text = re.sub(r"\+", " + ", text)
	text = re.sub(r"\-", " - ", text)
	text = re.sub(r"\=", " = ", text)
	text = re.sub(r"'", " ", text)
	text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
	text = re.sub(r":", " : ", text)
	text = re.sub(r" e g ", " eg ", text)
	text = re.sub(r" b g ", " bg ", text)
	text = re.sub(r" u s ", " american ", text)
	text = re.sub(r"\0s", "0", text)
	text = re.sub(r" 9 11 ", "911", text)
	text = re.sub(r"e - mail", "email", text)
	text = re.sub(r"j k", "jk", text)
	text = re.sub(r"\s{2,}", " ", text)

	return text

# 1. Split text, stem words, clean the text
train_df['question1'] = [filter(question1) for question1 in train_df['question1']]
train_df['question2'] = [filter(question2) for question2 in train_df['question2']]
test_df['question1'] = [filter(question1) for question1 in test_df['question1']]
test_df['question2'] = [filter(question2) for question2 in test_df['question2']]


# 2. TF-IDF 
train_df['q1_feats'] = tfidf(train_df['question1'])
train_df['q2_feats'] = tfidf(train_df['question2'])
test_df['q1_feats'] = tfidf(test_df['question1'])
test_df['q2_feats'] = tfidf(test_df['question2'])


# 3. Calculate distances (Euclidean)
train_longest = longest_question(train_df['q1_feats'], train_df['q2_feats'])
test_longest = longest_question(test_df['q1_feats'], test_df['q2_feats'])

train_questions1 = sequence.pad_sequences(train_df['q1_feats'], train_longest)
train_questions2 = sequence.pad_sequences(train_df['q2_feats'], train_longest)

test_questions1 = sequence.pad_sequences(test_df['q1_feats'], test_longest)
test_questions2 = sequence.pad_sequences(test_df['q2_feats'], test_longest)

train_distances = distance(train_questions1, train_questions2)
test_distances = distance(test_questions1, test_questions2)


# 4. Create 2D array 
train_distances = np.reshape(train_distances, (-1, 1))
test_distances = np.reshape(test_distances, (-1, 1))

# 5. Save TDIDF + Euc. distance to file 
np.save('train_distances', train_distances)
np.save('test_distances', test_distances)


# 4. XGBoost
X_train, X_test, y_train, y_test = train_test_split(train_distances, train_duplicate['is_duplicate'].values, test_size=0.2, random_state=0)


dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_test, label=y_test)

watchlist = [(dvalid, 'valid'), (dtrain, 'train')]

params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.02
params['max_depth'] = 4

num_round = 2

bst = xgb.train(params, dtrain, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)
bst.save_model('0001.model')

d_test = xgb.DMatrix(test_distances)
p_test = bst.predict(d_test)

sub = pd.DataFrame()
sub['test_id'] = test_df['test_id']
sub['is_duplicate'] = np.round(p_test).astype(int)
sub.to_csv('simple_xgb.csv', index=False)