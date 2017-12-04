import numpy as np
import spacy
from tqdm import tqdm

nlp = spacy.load('en')

def vec_features(row, word2tfidf):
    
    vecs = []
    for question in tqdm(list(row)): 
        doc = nlp(question)
        #print('doc 0 length: ', len(doc[0]))
        mean_vec = np.zeros([len(doc), 384])
    
        for word in doc: 
            # Word2vec
            vec = word.vector

            try: 
                idf = word2tfidf[str(word)]
            except: 
                idf = 0

            #print('vec shape size: ', vec.shape) # 384
            #print('idf shape size: ', idf)
            #print('mean vec: ', mean_vec)
            #print('mean vec: ', mean_vec.shape)

            mean_vec += vec * idf 
  
        mean_vec = mean_vec.mean(axis = 0)
        vecs.append(mean_vec)

        #print('lengths waka vec: ', vecs)

    return list(vecs)