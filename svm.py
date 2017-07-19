# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 13:54:40 2017

@author: Chandni J Purohit
"""

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import main_assn3 as ms3

excelfilename = r"C:\Users\Khushi\Downloads\Assignment_4_Data_and_Template.xlsx";
Data = pd.read_excel(excelfilename,sheetname="Training Data")

Target = Data['Failure']
Target6 = Data['Type']
Features = Data[Data.columns[0:15]]
x = np.ones((6600,1))
Features.insert(0,'x',x)


Mu,z,c,v,p,r,rec = ms3.get_mu_zcvpr(Features,2)
clf = SVC(kernel='linear')
clf.fit(Features, Target6)
#clf = OneVsRestClassifier(SVC(kernel='linear', probability=True, class_weight='auto'),n_jobs = -1)

#proba = clf.predict_proba(Features)
#Mu,z,c,v,p,r,rec = ms3.get_mu_zcvpr(Features,2)
from sklearn.decomposition import PCA
from sklearn import cross_validation

X_train, X_test, y_train, y_test = cross_validation.train_test_split(Features, Target6, test_size=0.4, random_state=0)

pca = PCA(n_components=2)
pca.fit(X_train)


x_train = pca.transform(X_train)
x_test = pca.transform(X_test)


clf = SVC(probability=True)
clf.fit(x_train,y_train)
print 'score', clf.score(x_train,y_train)
print 'score', clf.score(x_test,y_test)
print 'pred label', clf.predict(x_test)
