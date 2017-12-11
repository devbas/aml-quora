import numpy as np 
from utils.euclidean_distance import euclidean_distance
from scipy import spatial

def distance(questions1, questions2): 

	# We suppose both variables have the same len()
	distances = np.zeros(len(questions1))

	counter = 0
	for q1, q2 in zip(questions1, questions2): 
		distance = 1 - spatial.distance.cosine(q2, q1)
		#distance = euclidean_distance(q2, q1)
		distances[counter] = distance
		counter += 1

	return distances