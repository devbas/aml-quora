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
from keras.models import Sequential
from keras.layers import Dense, Activation
from gensim.models import word2vec
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

# 2. User setting 1: Import data
# Important: data needs to be stored in directory 'data' in parent folder of current working directory

path = os.getcwd()
os.chdir(path)
train_df = pd.read_csv("data/train_data.csv", nrows=500000, delimiter=',')
test_df = pd.read_csv("data/test_data.csv", nrows=100000, delimiter=',')
train_labels = pd.read_csv("data/train_labels.csv", nrows=500000, delimiter=',')
train_labels = train_labels['is_duplicate']
#train_df.head()

# 3. Split text
def word_tokens(row): 
   
  output = row;
  output['question1'] = word_tokenize(str(row['question1']).lower())
  output['question2'] = word_tokenize(str(row['question2']).lower())

  return output

train_tokenized = train_df.apply(word_tokens, axis=1, raw=True)
test_tokenized = test_df.apply(word_tokens, axis=1, raw=True)


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

test_tokenized = test_tokenized.apply(stemming_row, axis=1, raw=True)

# 6. Set vocabulary/Dictionary 
sentences = [train_tokenized['question1'], train_tokenized['question2']];
test_sentences = [test_tokenized['question1'], test_tokenized['question2']];



num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

print('sentences: ', sentences);

train_model = word2vec.Word2Vec(sentences, size=300, window = context, min_count=2, workers=num_workers, sample=downsampling)
test_model = word2vec.Word2Vec(test_sentences, size=300, window = context, min_count=2, workers=num_workers, sample=downsampling)



train_model.save("train_model")
test_model.save("test_model")
trainWordModel = word2vec.Word2Vec.load(train_model)
testWordModel = word2vec.Word2Vec.load(test_model)

#print('similarity-test: ',wordModel.wv.most_similar(positive=['muslim'], negative=['man']))

# 7. TF IDF


# 8. RNN (LSTM)
def trainNeuralNet(x_train, y_train):
  model = Sequential()
  model.add(Dense(64, input_dim=len(wordModel), activation='relu'))
  model.add(Dense(16, activation='relu'))
  model.add(Dense(16, activation='relu'))
  model.add(Dense(16, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))

  model.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['accuracy'])

  model.fit(x_train, y_train, epochs=10, batch_size=32)

def testNeuralNet(x_test):
  model = Sequential()
  model.add(Dense(64, input_dim=len(wordModel), activation='relu'))
  model.add(Dense(16, activation='relu'))
  model.add(Dense(16, activation='relu'))
  model.add(Dense(16, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))

  model.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['accuracy'])

  model.predict(x_test, batch_size=32)



trainNeuralNet(trainWordModel, train_labels)
testNeuralNet(testWordModel)