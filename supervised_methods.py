from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.model_selection import train_test_split
from statistics import mean
import nltk
from nltk.tokenize import TweetTokenizer
from csv import DictReader
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sentence_transformers import SentenceTransformer
from nltk.stem import PorterStemmer

nltk.download('stopwords') # get list of stopwords

def load_data(fn):
	""" Load csv and return an array of tweets and labels """
	DATA = []
	with open(fn,'r') as allTweetsFile:
		csv_dict_reader = DictReader(allTweetsFile)
		for oneTweet in csv_dict_reader:
			DATA.append([oneTweet['text'],oneTweet['labels']])
		allTweetsFile.close()

	return DATA

def sklearn_knn_predict(trainX, trainy, testX, distance_metric, k):
	"""Method used to run knn method"""
	knn_model = KNeighborsClassifier(algorithm = 'brute',n_neighbors=k, metric=distance_metric)
	training_model = knn_model.fit(trainX, trainy)
	predicts = training_model.predict(testX)

	return predicts

def knn_hyper(corpus, labels):
	"""Method used to grid search for best number of neighbors and best metric and plots results"""
	euclidean_accuracy = []
	manhattan_accuracy = []
	k_values = [1,3,5,7,9,11,13,15,17,19]
	for k in k_values:
		predict_euclid = sklearn_knn_predict(X_train, y_train, X_test, "euclidean", k)
		euclid_accuracy = round(get_accuracy(y_test, predict_euclid), 3)
		euclidean_accuracy.append(euclid_accuracy)

		predict_man = sklearn_knn_predict(X_train, y_train, X_test, "manhattan", k)
		man_accuracy = round(get_accuracy(y_test, predict_man), 3)
		manhattan_accuracy.append(man_accuracy)

	plt.plot(k_values, euclidean_accuracy, label = "euclidean")
	plt.plot(k_values, manhattan_accuracy, label = "manhattan")
	plt.xticks(k_values,k_values)
	plt.title("kNN validation accuracy for different values of k")
	plt.legend()
	plt.show()

def get_accuracy(y_true, y_predicted):
    """returns the fraction of correct predictions in y_predicted compared to y_true"""

    total_num = len(y_true)
    num_correct = 0
    for index in range(len(y_true)):
    	if y_true[index]==y_predicted[index]:
    		num_correct = num_correct + 1

    return num_correct/total_num

#Preprocessing text
def remove_stop_words(corpus):
	"""returns array of tweets that do not contain stop words"""
	stopwords = nltk.corpus.stopwords.words('english')
	tweets = []
	for c in corpus:
		text = ""
		for w in c.split():
			if w not in stopwords:
				text += w.lower()
		tweets.append(text)

	return tweets

def lemmatize_words(corpus):
	"""returns array of tweets with lemmatized words"""
	lem = nltk.stem.wordnet.WordNetLemmatizer()
	tweets = []
	for c in corpus:
		text = ""
		for w in c.split():
			text += lem.lemmatize(w.lower())
		tweets.append(text)

	return tweets

def stem_words(corpus):
	"""returns array of tweets with stemmed words"""
	ps = PorterStemmer()
	tweets = []
	for c in corpus:
		text = ""
		for w in c.split():
			text += ps.stem(str(w.lower()))
		tweets.append(text)

	return tweets


####################Embeddings Methods##########################
def sbert(data):
	"""Uses sBert transformer to vectorize each tweet and returns array of embeddings"""
	sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
	vec = sbert_model.encode(data)

	return vec

def tfidf(data):
	"""Uses tdidf to vectorize tweets and returns array of embeddings"""
	vectorizer = TfidfVectorizer()
	vec = vectorizer.fit_transform(data).toarray()

	return vec

def main():
	data = load_data("supervised_clean.csv")

	corpus = []
	labels = []

	#separates tweets and labels into two different arrays
	for tweet in data:
		corpus.append(tweet[0])
		labels.append(tweet[1])


	#pre processing steps to be uncommented when needing to be used
	#corpus = remove_stop_words(corpus)
	#corpus = lemmatize_words(corpus)
	#corpus = stem_words(corpus)

	#vectorization methods
	#sbert
	vectorizations = []
	X = sbert(corpus)
	vectorizations.append(X)

	#tf-idf
	X = tfidf(corpus)
	vectorizations.append(X)

	#goes through both vectorization methods
	vec_names = ["sbert","tf-idf"]
	num_vec = 0
	for v in vectorizations:
		#tries all models with sbert and then tfidf
		X_train, X_test, y_train, y_test = train_test_split(v, labels, train_size=0.70, test_size=0.30)
		print(vec_names[num_vec])
		num_vec += 1

		print("KNN")
		model = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
		scores = cross_validate(model, v, labels, cv=5,
			scoring=['f1_weighted', 'precision_weighted','recall_weighted','accuracy'])
		print("Accuracy:",mean(scores['test_accuracy']))
		print("Precision:",mean(scores['test_precision_weighted']))
		print("Recall:",mean(scores['test_recall_weighted']))
		print("F1-Score:",mean(scores['test_f1_weighted']))


		print("Perceptron")
		ppn = Perceptron(max_iter=10)
		scores =  cross_validate(ppn, v, labels, cv=5,
			scoring=['f1_weighted', 'precision_weighted','recall_weighted','accuracy'])
		print("Accuracy:",mean(scores['test_accuracy']))
		print("Precision:",mean(scores['test_precision_weighted']))
		print("Recall:",mean(scores['test_recall_weighted']))
		print("F1-Score:",mean(scores['test_f1_weighted']))
		
		print("SVM")
		clf = svm.SVC()
		scores =  cross_validate(clf, v, labels, cv=5,
			scoring=['f1_weighted', 'precision_weighted','recall_weighted','accuracy'])
		print("Accuracy:",mean(scores['test_accuracy']))
		print("Precision:",mean(scores['test_precision_weighted']))
		print("Recall:",mean(scores['test_recall_weighted']))
		print("F1-Score:",mean(scores['test_f1_weighted'])) 

		print("Neural Network")
		clf = MLPClassifier(hidden_layer_sizes=(100,100))
		scores =  cross_validate(clf, v, labels, cv=5,
			scoring=['f1_weighted', 'precision_weighted','recall_weighted','accuracy'])
		print("Accuracy:",mean(scores['test_accuracy']))
		print("Precision:",mean(scores['test_precision_weighted']))
		print("Recall:",mean(scores['test_recall_weighted']))
		print("F1-Score:",mean(scores['test_f1_weighted']))


main()