from sklearn.feature_extraction.text import TfidfVectorizer

from utils.vec_features import vec_features

def tfidf(questions): 
	tfidf = TfidfVectorizer(lowercase=False)
	tfidf.fit_transform(questions)

	word2tfidf = dict(zip(tfidf.get_feature_names(), tfidf.idf_)) # To research

	print('Features to vec')
	output = vec_features(questions, word2tfidf)
	#train_df['q2_feats'] = vec_features(train_df['question2'], word2tfidf)

	return output