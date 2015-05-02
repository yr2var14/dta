import csv
from sklearn import tree
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split

data = list(csv.DictReader(open('adult.csv','rU')))

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

def clean_data(data):
	data = remove_missing(data)
	data,target = refine_data(data)
	return data , target

data , target = clean_data(data)	

#print data

vec = DictVectorizer()
data = vec.fit_transform(data).toarray()

#print data.shape

data_train , data_test , target_train , target_test = train_test_split(data , target , test_size = 0.4)

t=tree.DecisionTreeClassifier()
t.fit(data_train , target_train)

print "Score of decision tree classifier algorithm :" , t.score(data_test,target_test)