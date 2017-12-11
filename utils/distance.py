import numpy as np 
from utils.euclidean_distance import euclidean_distance
from scipy import spatial

def distance(questions1, questions2, method): 

	# We suppose both variables have the same len()
	distances = np.zeros(len(questions1))

	counter = 0
	for q1, q2 in zip(questions1, questions2): 
		

		distance = euclidean_distance(q2, q1) if method == 'euclidean' else 1 - spatial.distance.cosine(q2, q1)
		distances[counter] = distance
		#distances[counter] = np.zeros(2)
		#distances[counter][0] = 1 - spatial.distance.cosine(q2, q1) # Cosine Similarity
		#distances[counter][1] = euclidean_distance(q2, q1) # Euclidean distance
		counter += 1

	return distances


# TFIDF Vector question 1
# TFIDF Vector question 2
# 