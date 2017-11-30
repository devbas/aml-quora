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
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.preprocessing.sequence import pad_sequences
from gensim.models import word2vec
import logging
import src.stemming as stemming
import src.tokens as tokens
import src.longest as longest
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

# 2. User setting 1: Import data
# Important: data needs to be stored in directory 'data' in parent folder of current working directory

path = os.getcwd()
os.chdir(path)
train_df = pd.read_csv("data/train_data.csv", nrows=1000, delimiter=',')
test_df = pd.read_csv("data/test_data.csv", nrows=1000, delimiter=',')
train_labels = pd.read_csv("data/train_labels.csv", nrows=1000, delimiter=',')
train_labels = train_labels['is_duplicate']
#train_df.head()

# 3. Split text
train_tokenized = train_df.apply(tokens.word_tokens, axis=1, raw=True)
test_tokenized = test_df.apply(tokens.word_tokens, axis=1, raw=True)

# 5. Stemming 
train_stemmed = train_tokenized.apply(stemming.stemming_row, axis=1, raw=True)
test_stemmed = test_tokenized.apply(stemming.stemming_row, axis=1, raw=True)

# 6. Zeropadding

def zero_padding(sequences, max_length):
    return pad_sequences(sequences, maxlen=max_length, padding='post')

train_padded1 = zero_padding(train_stemmed.question1, longestWordLength)
train_padded2 = zero_padding(train_stemmed.question2, longestWordLength)

test_padded1 = zero_padding(test_stemmed.question1, longestWordLength)
test_padded2 = zero_padding(test_stemmed.question2, longestWordLength)


# 6. Set vocabulary/Dictionary 
sentences = np.concatenate(train_padded1, train_padded2, axis=1);
test_sentences = np.concatenate(test_padded1, test_padded2, axis=1);

num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

trainWordModel = word2vec.Word2Vec(sentences, size=5, window = context, min_count=1, workers=num_workers, sample=downsampling)
testWordModel = word2vec.Word2Vec(test_sentences, size=5, window = context, min_count=1, workers=num_workers, sample=downsampling)



trainWordModel.save("trainWordModel")
testWordModel.save("testWordModel")
trainWords = word2vec.Word2Vec.load("trainWordModel")
testWords = word2vec.Word2Vec.load("testWordModel")

x_train=trainWords[trainWords.wv.vocab]
x_test=testWords[testWords.wv.vocab]

print('xtrain_length:', len(x_train))
print('ytrain_length:', len(train_labels))

#print('similarity-test: ',wordModel.wv.most_similar(positive=['muslim'], negative=['man']))

# 7. TF IDF


# 8. RNN (LSTM)
def trainNeuralNet(x_train, y_train):
  model = Sequential()
  model.add(Dense(64, input_dim=5, activation='relu'))
  model.add(Dense(16, activation='relu'))
  model.add(Dense(16, activation='relu'))
  model.add(Dense(16, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))

  model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

  model.fit(x_train, y_train, epochs=10, batch_size=32)

trainNeuralNet(sentences, train_labels)

#def testNeuralNet(x_test):
model = Sequential()
model.add(Dense(64, input_dim=5, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

output = model.predict(x_test, batch_size=32)


submission_df = pd.DataFrame(index=test_df.test_id, columns=['is_duplicate'], dtype=np.uint)
submission_df.index.name = 'test_id'
submission_df.is_duplicate = output

submission_df.to_csv('data/submission.csv')


#testNeuralNet(testWordModel)