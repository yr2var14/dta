
# -*- coding: utf-8 -*-
"""

@author: RibhavSoni
"""

import pandas as pd
import numpy as np
import pylab as P
import matplotlib.pyplot as pyplot

from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree


def readData(fileName):
    
    # Reading the data through Pandas package's own csv reader into a data frame
    
    dataFrame = pd.read_csv(fileName, header=0)
    
    return dataFrame


def cleanData(dataFrame):
    
    # Cleaning the data: filling missing values and coverting some variables to categorical variables:
    
    # Mapping Female to 0 and Male to 1 using a dictionary, to be able to apply scikit-learn's machine learning algorithms on the data:
    
    dataFrame['Gender'] = dataFrame['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    # All missing ages : Let them be the median of all ages:
    
    dataFrame.Age[ dataFrame.Age.isnull() ] = dataFrame.Age.dropna().median()
    
    # All missing Embarked values : Let them Embark from the most common place. There are only 2 missing values so that doesn't make much difference.
    
    dataFrame.Embarked[ dataFrame.Embarked.isnull() ] = dataFrame.Embarked.dropna().mode().values
    
    # Converting all Embark strings to int:
    
    dataFrame.Embarked = dataFrame.Embarked.map( {'S':1, 'Q':2, 'C':3}).astype(int)
    
    # The effect of "Siblings & Spouses" and "Parents and children" can be taken together as one 'Family_size' variable:
    
    dataFrame['Family_size'] = dataFrame['Parch'] + dataFrame['SibSp']
    
    # Remove the irrelevant columns : Name, column, Cabin, Ticket, Sex, Parch and Sibsp:
    
    dataFrame = dataFrame.drop(['Name', 'Sex', 'SibSp', 'Ticket', 'Cabin', 'PassengerId', 'Parch'], axis=1)
    
    return dataFrame


def printVisualizations(dataFrame):
    
    # Converting back to a numpy array, since we cannot apply scikit-learn's algorithms to a data frame object:
    
    train_data = dataFrame.values
    
    # Some analyses of the data:
    
    number_passengers = np.size(train_data[0::,0].astype(np.float))
    #print 'Number of Passengers : ', number_passengers
    
    number_survived = np.sum(train_data[0::,0].astype(np.float))
    #print 'Number of Survivors : ', int(number_survived)
    
    proportion_survivors = number_survived / number_passengers
    
    print '\nPercentage of people survived : ', "%.2f" %(100*proportion_survivors)
    
    women_data = dataFrame[dataFrame['Gender']==0].values
    
    men_data = dataFrame[dataFrame['Gender']==1].values
    
    print '\nPercentage of women on board', "%.2f" %(100*float(women_data.size) / float(train_data.size))
    
    print 'Percentage of men on board', "%.2f" %(100*float(men_data.size) / float(train_data.size))
    
    
    print '\nPercentage of women survived : ', "%.2f" %(100*np.sum(women_data[:,0]) / np.size(women_data[:,0]))
    
    print 'Percentage of men survived : ', "%.2f \n" %(100*np.sum(men_data[:,0]) / np.size(men_data[:,0]))
    
    
    # Survival Percentage from each class :
    
    classes = []
    
    for i in range(3):
        classes.append (dataFrame[dataFrame['Pclass']==i+1].values)
    
    survival_percentage = []
    
    for i in range(3):
        survival_percentage.append ( 100 * np.sum(classes[i][:,0]) / np.size(classes[i][:,0]) )
    
    ages=[]
    aa=[]    
    n_age=[]
    for i in range (16):
        aa.append( 5*i + 2.5 )
        df = dataFrame[ ( dataFrame['Age'].astype(int) >= 5*i ) & ( dataFrame['Age'].astype(int) < 5*(i+1) ) ].values
        ages.append( 100 * np.sum(df[:,0]) / np.size(df[:,0]) )
        n_age.append(np.size(df[:,0]))
    
    
    # Some visualizations of the data:
    
    
    
    # No. of members of each class:
    
    dataFrame['Pclass'].hist(bins=3, range=(1,3), alpha = .5)
    P.title(' Number of passengers of each class ')
    P.ylabel(' Number of people ')
    P.xlabel(' Class ')
    P.show()
    
    # Survival Rate of different classes:
    
    pyplot.title(' Survival Rate with Passenger Class ')
    pyplot.xlabel(' Passenger Class ')
    pyplot.ylabel(' Survival Rate')
    pyplot.bar( [ 1, 2, 3 ] , survival_percentage)
    pyplot.show()
    
    # No. of people travelling with family members :
    
    dataFrame['Family_size'].hist(bins=10, range=(0,10), alpha = .5)
    P.title(' No. of people travelling with family members ')
    P.ylabel('Number of people')
    P.xlabel('Famly Size')
    P.show()
    
    # No. of passengers of each age group on board :
    
    pyplot.title(' Number of passengers in each age group ')
    pyplot.xlabel(' Age ')
    pyplot.ylabel(' Number of passengers ')
    pyplot.bar(aa, n_age)
    pyplot.show()    
    
    # Survival Chances with age of a person:
    
    pyplot.title(' Survival Chances with Age')
    pyplot.xlabel(' Age ')
    pyplot.ylabel(' Survival Rate ')
    pyplot.bar(aa, ages)
    pyplot.show()    
    
    
    
def visualizeRandomForestClassifier(X_train, y_train, X_test, y_test):

    x = []
    y = []
    for i in range(1, 50):
        x.append(2*i)
        z = []
        # Taking the average of 10 values for each value of no. of estimators:
        for j in range(10):
            z.append(RandomForestClassifier(n_estimators=2*i).fit(X_train,y_train).score(X_test, y_test))
        y.append(float(sum(z))/len(z) if len(z) > 0 else float('nan'))
        
    pyplot.title('      Accuracy of Random Forest Classifier with no. of estimators ')
    pyplot.ylabel(' Score ')
    pyplot.xlabel(' Number of estimators ')
    pyplot.plot(x, y)


def trainModel(modelName, X_train, y_train):
    
    
    if modelName=='LinearSVM':
        
        # Applying Linear SVM Classifier model to the training data:
        
        linearSVMClassifierObject = SVC(kernel='linear', C=1).fit(X_train, y_train)
        
        return linearSVMClassifierObject
    
    elif modelName=='DecisionTree':
        
        # Applying Decision Tree Classifier model to the training data:
        
        decisionTreeClassifierObject = tree.DecisionTreeClassifier().fit(X_train,y_train)
        
        return decisionTreeClassifierObject    
        
    elif modelName=='RandomForest':
        
        # Applying Random Forest Classifier model to the training data:
    
        randomForestCLassifierObject = RandomForestClassifier(n_estimators = 200).fit(X_train,y_train)
        
        return randomForestCLassifierObject
        

def getClassifierScore(ClassifierObject, X_test, y_test):
    
    return ClassifierObject.score(X_test, y_test)
    
    
def main():
    
    trainDataFrame = readData('train.csv')
    
    train_df = cleanData(trainDataFrame)

    printVisualizations(train_df)
    
    train_data = train_df.values
    
    # Using cross-validation to check the performance of our models by reserving a part (40%) of the data for testing
    
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_data[0::,1::], train_data[0::,0], test_size=0.4, random_state=0)
        
    print 'Training Linear SVM model ... \n'
    
    Classifier1 = trainModel('LinearSVM', X_train, y_train)
    
    print '\nAccuracy of linear SVM model : ', getClassifierScore(Classifier1, X_test, y_test), '\n\n'

    print 'Training Decision Tree Classifier model ... \n'
    
    Classifier2 = trainModel('DecisionTree', X_train, y_train)
    
    print '\nAccuracy of Decision Tree Classifier : ', getClassifierScore(Classifier2, X_test, y_test), '\n\n'
    
    print 'Training Random Forest Classifier model ... \n'
    
    Classifier3 = trainModel('RandomForest', X_train, y_train)
    
    print '\nAccuracy of Random Forest Classifier : ', getClassifierScore(Classifier3, X_test, y_test), '\n'
    
    print '\n\n Variation in Accuracy of Random Forest Classifier with no. of estimators : \n'
    
    visualizeRandomForestClassifier(X_train, y_train, X_test, y_test)
    
    
if __name__ == '__main__':
    
    main()
    
    
