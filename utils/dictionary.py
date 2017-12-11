dictionary = dict()

def word_to_index(word): 
	if word not in dictionary.values(): 
		dictionary.append(word)

	print('dictionary: ', dictionary);
	return dictionary[word]


def create_dict(row): 
	print('row: ', row['question1'])

	for count, word in enumerate(row['question1']):
		row['question1'][count] = word_to_index(word)

	for count, word in enumerate(row['question2']): 
		row['question2'][count] = word_to_index(word)

	return row
