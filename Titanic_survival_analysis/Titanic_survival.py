import pandas as pd
import numpy as np
import pylab as P
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

# Reading the data through Pandas package's own csv reader into a data frame
train_df = pd.read_csv('train.csv', header=0)

# Cleaning the data: filling missing values and coverting some variables to categorical variables:

# Mapping Female to 0 and Male to 1 using a dictionary, to be able to apply scikit-learn's machine learning algorithms on the data:
train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# All missing ages : Let them be the median of all ages:
train_df.Age[ train_df.Age.isnull() ] = train_df.Age.dropna().median()

# All missing Embarked values : Let them Embark from the most common place. There are only 2 missing values so that doesn't make much difference.
train_df.Embarked[ train_df.Embarked.isnull() ] = train_df.Embarked.dropna().mode().values

# Converting all Embark strings to int:
train_df.Embarked = train_df.Embarked.map( {'S':1, 'Q':2, 'C':3}).astype(int)

# The effect of "Siblings & Spouses" and "Parents and children" can be taken together as one 'Family_size' variable:
train_df['Family_size']=train_df['Parch']+train_df['SibSp']

# Remove the irrelevant columns : Name, column, Cabin, Ticket, Sex, Parch and Sibsp:
train_df = train_df.drop(['Name', 'Sex', 'SibSp', 'Ticket', 'Cabin', 'PassengerId', 'Parch'], axis=1)

# Convert back to a numpy array
train_data = train_df.values

number_passengers = np.size(train_data[0::,0].astype(np.float))
#print 'Number of Passengers : ', number_passengers

number_survived = np.sum(train_data[0::,0].astype(np.float))
#print 'Number of Survivors : ', int(number_survived)

proportion_survivors = number_survived / number_passengers
print '\nPercentage of people survived : ', "%.2f" %(100*proportion_survivors)

women_data = train_df[train_df['Gender']==0].values
men_data = train_df[train_df['Gender']==1].values

print '\nPercentage of women on board', "%.2f" %(100*float(women_data.size)/float(train_data.size))
print 'Percentage of men on board', "%.2f" %(100*float(men_data.size)/float(train_data.size))

print '\nPercentage of women survived : ', "%.2f" %(100*np.sum(women_data[:,0])/np.size(women_data[:,0]))
print 'Percentage of men survived : ', "%.2f \n" %(100*np.sum(men_data[:,0])/np.size(men_data[:,0]))

# Survival Percentage from each class :
classes = []
for i in range(3):
    classes.append (train_df[train_df['Pclass']==i+1].values)

survival_percentage = []
for i in range(3):
    survival_percentage.append ( 100 * np.sum(classes[i][:,0])/np.size(classes[i][:,0]) )

for i in range(3):
      print 'Survival rate of people from class %d : %.2f' %(i+1,survival_percentage[i])

# Some visualizations of the data:

# No. of people travelling with family members :
train_df['Family_size'].hist(bins=10, range=(0,10), alpha = .5)
P.title(' No. of people travelling with family members ')
P.ylabel('Number of people')
P.xlabel('Famly Size')
P.show()

# No. of members of each class:
train_df['Pclass'].hist(bins=3, range=(1,3), alpha = .5)
P.title(' Class of passengers ')
P.ylabel(' Number of people ')
P.xlabel(' Class ')
P.show()


# Using cross-validation to check the performance of our models by reserving a part (40%) of the data for testing
X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_data[0::,1::], train_data[0::,0], test_size=0.4, random_state=0)

clf1 = SVC(kernel='linear', C=1).fit(X_train, y_train)
print '\nAccuracy of linear SVM model : ', clf1.score(X_test, y_test)

clf2 = tree.DecisionTreeClassifier().fit(X_train,y_train)
print '\nAccuracy of Decision Tree Classifier : ', clf2.score(X_test, y_test)

clf3 = RandomForestClassifier(n_estimators=100).fit(X_train,y_train)
print '\nAccuracy of Random Forest Classifier : ', clf3.score(X_test, y_test), '\n'

