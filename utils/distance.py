import numpy as np 
from utils.euclidean_distance import euclidean_distance
from scipy import spatial

def distance(questions1, questions2, method): 

	counter = 0
	for q1, q2 in zip(questions1, questions2): 

		# We currently support two distances: euclidean and cosine similarity. This explains the shorthand. 
		distance = euclidean_distance(q2, q1) if method == 'euclidean' else 1 - spatial.distance.cosine(q2, q1)
		distances[counter] = distance
		counter += 1

	return distances
