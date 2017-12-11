from nltk.stem.porter import *

stemmer = PorterStemmer()

def stemming_row(row): 

  return [stemmer.stem(w) for w in row] 


