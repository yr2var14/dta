import csv
import copy
import random
import pandas as pd
import pylab as P
import nbmodule as x
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
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


def print_dataStats():

	df = pd.read_csv('adult.csv',header = 0)
	
	#visualisation for age
	df['age'].hist(bins=16,range=(0,80),alpha=.5)
	P.show()
	
	#visualisation for education-level
	df['education-num'].hist(bins=16 , range=(0,16))
	P.show()


def train_model(model_name , data_train , target_train):
	
	if(model_name == 'NearestNeighbor'):
		print "Fitting the nearest neighbor model..."
		n=KNeighborsClassifier(n_neighbors=20)
		n.fit(data_train , target_train)
		return n;

	elif(model_name == 'DecisionTree'):
		print "Fitting the decision tree model..."
		d=tree.DecisionTreeClassifier()
		d.fit(data_train , target_train),0
		return d;


def main():
	
	#using DictVectorizer to get data in a Scikit-Learn-usable form 
	vec = DictVectorizer()

	print "Loading training set..."
	data = list(csv.DictReader(open('adult.csv','rU')))
	data_refined , target = clean_data(data)
	data_refined= vec.fit_transform(data_refined).toarray() 

	print "Loading test set..."
	data = list(csv.DictReader(open('test.csv','rU')))
	data_test , target_test = clean_data(data)
	data_test = vec.fit_transform(data_test).toarray()

	#plotting data for visualizations
	print_dataStats()

	
	data_train , cv_data_test , target_train , cv_target_test = train_test_split( data_refined , target, test_size = 0.33)

	# decision tree

	d=train_model('DecisionTree' ,data_train ,target_train)
	print "Score of decision tree algorithm on cross-validation set:" , float(d.score(cv_data_test,cv_target_test))*100,"100"
	print "Score of decision tree algorithm on test set:" , float(d.score(data_test, target_test))*100 ,"%"

	# k-Nearest Neighbors
	
	n=train_model('NearestNeighbor' ,data_train ,target_train)
	print "Score of nearest neighbour algorithm on cross-validation set:" , float(n.score(cv_data_test,cv_target_test))*100,"%"
	print "Score of nearest neighbour algorithm on test set:" , float(n.score(data_test, target_test))*100 ,"%"

	
main()
