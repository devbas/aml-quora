def maximum(questions): 
	return questions.map(lambda x: len(x)).max()

def longest_question(question1, question2):
	return max(
		maximum(question1),
        maximum(question2)
	)