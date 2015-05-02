import copy
import random
from sklearn.feature_extraction import DictVectorizer

def divide_attributes(data):
	"""
	function to divide data in two subsets , one having continuous value type attributes to apply Gaussian Naive Bayes and the other having categorical type attributes so as to apply Multinomial Naive Bayes
	"""
	continuous=['fnlwgt','age','education-num','capital-gain','capital-loss','hours-per-week']
	categorical=['workclass','education','marital-status','occupation','relationship','race','sex','native-country']
	
	gaussian_dict=copy.deepcopy(data)
	multinomial_dict=copy.deepcopy(data)

	for i in range(0,len(data)):
		for key in continuous:
			del multinomial_dict[i][key]	
		for key in categorical:
			del gaussian_dict[i][key]

	vec = DictVectorizer()
	gaussian_dict = vec.fit_transform(gaussian_dict).toarray()
	multinomial_dict = vec.fit_transform(multinomial_dict).toarray()
	return gaussian_dict , multinomial_dict

def cross_validate(gaussian_dict , multinomial_dict , target):
	trainSize = int(len(target) * 0.67)
	gaussian_dict_train = []
	multinomial_dict_train = []
	target_train=[]
	gaussian_dict_test = list(gaussian_dict)
	multinomial_dict_test = list(multinomial_dict)
	target_test = list(target)

	while len(target_train) < trainSize:
		index = random.randrange(len(target_test))
		gaussian_dict_train.append(gaussian_dict_test.pop(index))
		multinomial_dict_train.append(multinomial_dict_test.pop(index))
		target_train.append(target_test.pop(index))

	return gaussian_dict_train ,gaussian_dict_test , multinomial_dict_train , multinomial_dict_test , target_train , target_test 


def predict(Gaussian_fit ,Multinomial_fit):
	total_prob=[]
	prediction = []
	for i in range(len(Gaussian_fit)):
		a=[]
		a.append(Gaussian_fit[i][0]*Multinomial_fit[i][0])
		a.append(Gaussian_fit[i][1]*Multinomial_fit[i][1])
		total_prob.append(a)
		if total_prob[i][0]>total_prob[i][1]:
			prediction.append(0)
		else :
			prediction.append(1)
	return prediction

def score(prediction , true_labels):
	score = 0
	for i in range(0,len(true_labels)):
		if prediction[i] == true_labels[i]:
			score+=1
	score = float(score) / len(true_labels)
	return score
