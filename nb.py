import csv
import copy
import random
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer

data = list(csv.DictReader(open('adult.csv','rU')))
target = list()

#remove samples with missing values
for i in data:
	if ' ?' in i.values():
		data.pop()

continuous=['fnlwgt','age','education-num','capital-gain','capital-loss','hours-per-week']
categorical=['workclass','education','marital-status','occupation','relationship','race','sex','native-country']
	
#create list of labels
for i in data:
	if i['income'] ==' >50K':
		target.append(1)
	else:
		target.append(0)
	del i['income']
	for key in continuous:
		i[key]=int(i[key])

gaussian_dict=copy.deepcopy(data)
multinomial_dict=copy.deepcopy(data)

for i in range(0,len(target)):
	for key in continuous:
		del multinomial_dict[i][key]	
	for key in categorical:
		del gaussian_dict[i][key]

vec = DictVectorizer()
gaussian_dict = vec.fit_transform(gaussian_dict).toarray()
multinomial_dict = vec.fit_transform(multinomial_dict).toarray()

# splitting the dataset manually in the ratio of 2:1

trainSize = int(len(data) * 0.67)
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


g=GaussianNB()
g.fit(gaussian_dict_train, target_train)

m=MultinomialNB()
m.fit(multinomial_dict_train, target_train)

G=g.predict_proba(gaussian_dict_test)
M=m.predict_proba(multinomial_dict_test)

total_prob=[]
prediction = []

for i in range(len(target_test)):
	a=[]
	a.append(G[i][0]*M[i][0])
	a.append(G[i][1]*M[i][1])
	total_prob.append(a)
	if total_prob[i][0]>total_prob[i][1]:
		prediction.append(0)
	else :
		prediction.append(1)

def score(prediction , true_labels):
	score = 0
	for i in range(0,len(true_labels)):
		if prediction[i] == true_labels[i]:
			score+=1
	score = float(score) / len(true_labels)
	return score

print score(prediction , target_test)




