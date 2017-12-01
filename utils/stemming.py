from nltk.stem.porter import *

stemmer = PorterStemmer()

def stemming_row(row): 

  return [stemmer.stem(w) for w in row] 

  #output = row; 
  #output['question1'] = [stemmer.stem(w) for w in row['question1']]
  #output['question2'] = [stemmer.stem(w) for w in row['question2']]

  #return output


