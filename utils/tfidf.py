from sklearn.feature_extraction.text import TfidfVectorizer

from utils.vec_features import vec_features

def tfidf(questions): 
	tfidf = TfidfVectorizer(lowercase=False,sublinear_tf=True)
	tfidf.fit_transform(questions)

	word2tfidf = dict(zip(tfidf.get_feature_names(), tfidf.idf_)) # To research

	print('Features to vec')
	output = vec_features(questions, word2tfidf)

	return output