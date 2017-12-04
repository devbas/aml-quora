# 8. Validate model 
train_df = train_df.reindex(np.random.permutation(train_df.index))

# set number of train and test instances
num_train = int(train_df.shape[0] * 0.88)
num_validation = train_df.shape[0] - num_train 

print("Number of training pairs: %i"%(num_train))
print("Number of testing pairs: %i"%(num_validation))

# init data data arrays
X_train = np.zeros([num_train, 2, 300])
X_validation  = np.zeros([num_validation, 2, 300])
Y_train = np.zeros([num_train]) 
Y_validation = np.zeros([num_validation])






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