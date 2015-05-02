import csv
import numpy as np
import pandas as pd
import pylab as P
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.learning_curve import validation_curve

def remove_missing(data):
	for i in data :
		if ' ?' in i.values():
			data.pop()
	return data
			
def refine_data(data):	
	"""
	function to get data in usable form , i.e. , remove samples with missing values and convert continuous value attributes from strings to continuous forms 
	"""
	continuous=['fnlwgt','age','education-num','capital-gain','capital-loss','hours-per-week']
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

def main():
	print "Loading training set..."
	data = list(csv.DictReader(open('adult.csv','rU')))
	data = remove_missing(data)
	data_refined , target = refine_data(data)


	#using DictVectorizer to get data in a Scikit-Learn-usable form 
	vec = DictVectorizer()
	data_refined= vec.fit_transform(data_refined).toarray() 

	data_train , data_test , target_train , target_test = train_test_split( data_refined , target, test_size = 0.4)

	for k in [5,10,15,20,25,30,35,40]:

		n=KNeighborsClassifier(n_neighbors=k)
		n.fit(data_train , target_train)

		print "Score of nearest neighbour algorithm on cross-validation set for k=" ,k,"is :" ,n.score(data_test,target_test)

		data = list(csv.DictReader(open('test.csv','rU')))
		data = remove_missing(data)
		data_refined , target = refine_data(data)

		#using DictVectorizer to get data in a Scikit-Learn-usable form 
		vec = DictVectorizer()
		data_refined = vec.fit_transform(data_refined).toarray()

		print "Score of nearest neighbour algorithm on test set for k = ",k,"is :",float(n.score(data_refined, target))*100 ,"%"

main()
