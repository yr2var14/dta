import csv
import nbmodule as x
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer

continuous=['fnlwgt','age','education-num','capital-gain','capital-loss','hours-per-week']
categorical=['workclass','education','marital-status','occupation','relationship','race','sex','native-country']

def remove_missing(data):
	for i in data :
		if ' ?' in i.values():
			data.pop()
	return data
			

def refine_data(data):	
	"""
	function to get data in usable form , i.e. , remove samples with missing values and convert continuous value attributes from strings to continuous forms 
	"""
	target = list()	
	for i in data:
		for key in continuous:
			i[key]=int(i[key])
		if i['income'] ==' >50K':
			target.append(1)
		else:
			target.append(0)
		del i['income']
	return data , target

def clean_data(data):
	data = remove_missing(data)
	data,target = refine_data(data)
	return data , target


def main():
	
	print "Loading training set..."
	#Loading data in the form of list of dicts from the csv file of training set

	data = list(csv.DictReader(open('adult.csv','rU')))
	data_refined , target = clean_data(data)

	gaussian_dict , multinomial_dict = x.divide_attributes(data_refined)
	#splitting dataset into training set and cross-validation set
	gaussian_dict_train ,gaussian_dict_test , multinomial_dict_train , multinomial_dict_test , target_train , target_test = x.cross_validate(gaussian_dict , multinomial_dict , target)

	print "Running naive bayes on the training set"
	g=GaussianNB()
	g.fit(gaussian_dict_train, target_train)

	m=MultinomialNB()
	m.fit(multinomial_dict_train, target_train)

	G=g.predict_proba(gaussian_dict_test)
	M=m.predict_proba(multinomial_dict_test)

	prediction = x.predict(G , M)

	print "Score of the naive bayes algorithm on the cross-validation set is:" ,x.score(prediction , target_test)

	print "Loading test set..."
	#Loading data in the form of list of dicts from the csv file of training set

	data = list(csv.DictReader(open('test.csv','rU')))
	data = remove_missing(data)
	data_refined , target = refine_data(data)

	gaussian_dict , multinomial_dict = x.divide_attributes(data_refined)

	print "Running naive bayes on the test set"

	G=g.predict_proba(gaussian_dict)
	M=m.predict_proba(multinomial_dict)

	prediction = x.predict(G,M)

	print "Score of the naive bayes algorithm on the test set is:" , x.score(prediction , target)

main()



