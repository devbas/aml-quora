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
train_df = pd.read_csv("data/train_data.csv", nrows=100, delimiter=',')
test_df = pd.read_csv("data/test_data.csv", nrows=100, delimiter=',')
train_duplicate = pd.read_csv("data/train_labels.csv", nrows=100, delimiter=',')

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

	#text = stemmer.stem(text)

	#text = text.split()

	#text = text[:5]

	#print('text: ', text)

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


# 4. XGBoost
X_train, X_test, y_train, y_test = train_test_split(train_distances, train_duplicate['is_duplicate'].values, test_size=0.2, random_state=0)

print('train shape: ', X_train.shape)
print('train test shape: ', X_test.shape)
print('y train shape: ', y_train.shape)
print('y test shape: ', y_test.shape)

#print('dtrain: ', x_train.num_col())

#x_train = np.array(x_train).reshape((1,-1))
#y_train = np.array(y_train).reshape((1,-1))

dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_test, label=y_test)

#dtrain = xgb.DMatrix(np.array(X_train).reshape((1,-1)), label=y_train)
#dvalid = xgb.DMatrix(np.array(X_test).reshape((1,-1)), label=y_test)

print('dtrain: ', dtrain.num_col())

evallist = [(dvalid, 'eval'), (dtrain, 'train')]

params = {'max_depth':1, 'eta':1, 'silent':1, 'objective':'binary:logistic' }

num_round = 2

bst = xgb.train(params, dtrain, 2, evallist)
#print(log_loss(train.is_duplicate, bst.predict(xgb.DMatrix(df))))
'''bst.save_model('0001.model')

print('train_distances: ', bst)'''