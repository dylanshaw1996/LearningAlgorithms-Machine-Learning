#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @author:dylan.shaw@mycit.ie

# Name: Dylan Shaw 
#Student ID: R00128608


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets 
from sklearn import tree 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.svm import SVC
from sklearn import metrics

df = pd.read_csv('finance.csv', encoding = "ISO-8859-1")


def task1(): 
    flt = df[['age', 'job', 'y']].copy() 
    flt = flt[['age', 'job', 'y']].dropna() 
    allJobs = np.unique(flt['job']).astype(str) 
    dict2 = {} 
    c = 1 
    for j in allJobs: 
        dict2[j] = c
        c = c + 1
    flt['job'] = flt['job'].map(dict2) 
    allYs = np.unique(flt['y']).astype(str) 
    dict1 = {} 
    c = 1 
    for j in allYs: 
        dict1[j] = c 
        c = c + 1
    flt['y'] = flt['y'].map(dict1) 
    X = flt[['age', 'job']]
    y = flt[['y']]
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state=42)
    clf = DecisionTreeClassifier() 
    
    cv_results = cross_validate(clf, X, y, cv=10, scoring='accuracy', return_train_score=True)
    
    #print(cv_results['train_score'].mean())
    print('First Test Score : ',cv_results['test_score'].mean())
    
    #clf.fit(X_train, y_train) 
    #print('Training', clf.score(X_train, y_train)) 
    #print('Test', clf.score(X_test, y_test)) 
    
    print('')
    
    flt = df[['age', 'balance', 'y']].copy() 
    flt = flt[['age', 'balance', 'y']].dropna() 
     
    allYs = np.unique(flt['y']).astype(str) 
    dict1 = {} 
    c = 1 
    for j in allYs: 
        dict1[j] = c 
        c = c + 1
    flt['y'] = flt['y'].map(dict1) 
    X = flt[['age', 'balance']]
    y = flt[['y']]
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state=42)
    clf = DecisionTreeClassifier() 
    cv_results = cross_validate(clf, X, y, cv=10, scoring='accuracy', return_train_score=True)
    
    #print(cv_results['train_score'].mean())
    print('Second Test Score : ', cv_results['test_score'].mean())
    print('')
    
    #clf.fit(X_train, y_train) 
    #print('Training', clf.score(X_train, y_train)) 
    #print('Test', clf.score(X_test, y_test)) 
    
    print('As you can see from the results above, the first test score will give you a better accuracy when you use Age, Job & Y ')
    
def task2(): 
    flt = df [['age', 'balance']]
    flt = flt.fillna(flt.mean())
    
    min_max_scaler = preprocessing.MinMaxScaler()
    flt = min_max_scaler.fit_transform(flt) 
    
    #print(flt)
    #kalg = KMeans(n_clusters=2) 
    #kalg.fit(flt) 
    #print(kalg.labels_)
    
    cost = [] 
    
    for i in range(8): 
        kmeans = KMeans(n_clusters=i+1, random_state = 0).fit(flt)
        #print(kmeans.inertia_)
        #print(kmeans.cluster_centers_)
        cost.append(kmeans.inertia_)
    
    index = np.arange(1, 9)
    plt.plot(index, cost) 
    
    print('')
    
    print('Task 2: From the graph above you can see that the best number of clusters to use is 3, as per the eblow line on the graph. ')
    
def task3():
    newDataSet  = df[['age', 'job', 'education', 'loan', 'housing']].copy() 
    newDataSet = newDataSet[['age', 'job', 'education', 'loan', 'housing']].dropna()
    allJobs = np.unique(newDataSet['job']).astype(str) 
    dict1 = {} 
    c = 1 
    for j in allJobs: 
        dict1[j] = c
        c = c + 1
    newDataSet['job'] = newDataSet['job'].map(dict1)
    
    allEducation = np.unique(newDataSet['education']).astype(str)
    dict2 = {} 
    c = 1 
    for j in allEducation: 
        dict2[j] = c
        c = c + 1
    newDataSet['education'] = newDataSet['education'].map(dict2)
    
    allLoan = np.unique(newDataSet['loan']).astype(str)
    dict3 = {} 
    c = 1 
    for j in allLoan: 
        dict3[j] = c
        c = c + 1
    newDataSet['loan'] = newDataSet['loan'].map(dict3)
    
    allHousing = np.unique(newDataSet['housing']).astype(str)
    dict4 = {} 
    c = 1 
    for j in allHousing: 
        dict4[j] = c
        c = c + 1
    newDataSet['housing'] = newDataSet['housing'].map(dict4)
    
    X = newDataSet[['age', 'job', 'education', 'housing']]
    y = newDataSet[['loan']]
    clf = DecisionTreeClassifier()
    cv_results = cross_validate(clf, X, y, cv=10, scoring='accuracy', return_train_score=True)
    DTC = (cv_results['test_score'].mean())
    
    
    clf = RandomForestClassifier(n_estimators = 100 )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    clf.fit(X_train, y_train.values.ravel()) 
    pred = clf.predict(X_test)
    RFC = (metrics.accuracy_score(y_test, pred)) 
    
    
    knm = KNeighborsClassifier(n_neighbors=5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    knm.fit(X_train, y_train.values.ravel())
    y_pred = knm.predict(X_test) 
    KNC = (metrics.accuracy_score(y_test, y_pred))
    
    clf = svm.SVC(kernel='linear')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    clf.fit(X_train, y_train.values.ravel())
    y_pred = clf.predict(X_test)
    SVC = (metrics.accuracy_score(y_test, y_pred))
    
    gnb = GaussianNB()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    gnb.fit(X_train, y_train.values.ravel())
    y_pred = gnb.predict(X_test)
    GNB = (metrics.accuracy_score(y_test, y_pred))
    
    
    fig = plt.figure() 
    ax = fig.add_axes([0,0,1,1])
    testData = ['DTC', 'RFC', 'KNC', 'SVC', 'GNB']
    score = [DTC, RFC, KNC, SVC, GNB]
    ax.bar(testData, score) 
    plt.show() 
    
    print('')
    print('Task 3: As per the graph above, you can see that out of all the algorithims, SVC will give you the best accuracy available. ')
    print('')
    
    
def task4(): 
    jobList  = df[['job']].copy() 
    jobList = jobList[['job']].dropna()
    
    x = pd.value_counts(df['job']).plot(kind='bar')   
    print(x)
    
    
    print('')
    print('Task 4: As per the graph above, you can see that blue-collar is the most popular job and unknown is the least popular. ')
    print('')
    
def task5(): 
    ageList = df[['age']].copy() 
    ageList = ageList[['age']].dropna()
    
    x = pd.value_counts(df['age']).plot(kind='box')   
    print(x)
    
    print('')
    print('Task 5: As per the graph above, you can see what the outliars are in the data set. ')
    print('')
    
def task6(): 
    maritalList = df[['marital']].copy() 
    maritalList = maritalList[['marital']].dropna()
    
    x = pd.value_counts(df['marital']).plot(kind='pie', autopct='%1.0f%%')   
    print(x)
    print('')
    print('Task 6: As per the graph above, you can see what the marital percentages are. ')
    print('')

def main():               
    task1()
    task2()
    task3()
    task4() 
    task5() 
    task6() 
    
    
main() 


