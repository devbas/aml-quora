from nltk.tokenize import word_tokenize

def word_tokens(row): 
  
  return word_tokenize(row) 
  #output = row;
  #output['question1'] = word_tokenize(str(row['question1']).lower())
  #output['question2'] = word_tokenize(str(row['question2']).lower())

  #return output