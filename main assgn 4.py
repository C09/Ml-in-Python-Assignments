# -*- coding: utf-8 -*-
"""
Created on Sun Jul 09 18:31:16 2017

@author: Chandni J Purohit
"""

import pandas as pd
import numpy as np
from scipy import linalg as lg
import itertools
from sklearn.metrics import confusion_matrix as cm
#from keras.utils import np_utils
from sklearn.preprocessing import OneHotEncoder


def WClassifier(Xa_PI,Target):
    W = np.dot(Xa_PI,Target)
    return W

def main():
    
    #r is a raw string literal
    excelfilename = r"C:\Users\Khushi\Downloads\Assignment_4_Data_and_Template.xlsx";
    Data = pd.read_excel(excelfilename,sheetname="Training Data")
    TestingData=pd.read_excel(excelfilename,sheetname="To be classified",skiprows=3)
    TestingTarget = TestingData['Failure']
    TestingTarget6C = TestingData['Type']
    TestingData = TestingData[Data.columns[0:15]]
    xt = np.ones((50,1))   
    TestingData.insert(0,'x',xt)
    Target = Data['Failure']
    Features = Data[Data.columns[0:15]]
    x = np.ones((6600,1))
    Features.insert(0,'x',x)
    Feature_PI = lg.pinv(Features)
    te = OneHotEncoder(sparse=False)
    Target6 = Target6.reshape(-1,1)
    te.fit(Target6)
    W = WClassifier(Feature_PI,Target)
    BinaryClassifier = WClassifier(Features,W)
    cm_DF = cm(Target,np.sign(BinaryClassifier))
    PredictBinaryClassifier = WClassifier(TestingData,W) #Applying Binary Classifier
    
#    imported from excel using left right
#    Target6C = Data[Data.columns[17:22]] # this is correct
#    T6C = np.array(Target6C) # this is also currect
#    T6C[np.where(T6C ==0)] = -1

    te6 = te.transform(Target6)
    orgte6=te6
    #te6[np.where(te6 == 0)] = -1   #te6 is correct
    W6C = WClassifier(Feature_PI,te6)
    PredictMultiClassifier = WClassifier(Features,W6C)
    pmc2 = sign(PredictMultiClassifier)
    pmc2[np.where(pmc2==-1)]=0
    decoded2 = pmc2.dot(te.active_features_).astype(int)
    cm_DF6C = cm(Target6,decoded2)
    
    
    BinaryClassifier = WClassifier(TestingData,W)
    MultiClassifier = WClassifier(TestingData,W6C)
    
    
    
    TrClssfrData = pd.read_excel(excelfilename,sheetname="Training Data2")
    cm_DF6C = cm(T6C,np.sign(TrClassifier))
    PredictMultiClass = WClassifier(TestingData,W6C)
    

#    series =  list(itertools.product([0,1],repeat=6))
#    series = series[0:6]
#    trydf = pd.DataFrame(Target6C)
#    df = pd.DataFrame({'A': [0, 4, 5, 6, 7, 7, 6,5]})

#    
#    
#    And the mapping that you desire:
        
    dec2bin = lambda x : '%06.0f' % int(bin(x)[2:])
    df = pd.DataFrame(Target6C)
    df['binary'] = df["Type"].apply(dec2bin)
    
        
#    for i in range(0,5):
    mapping = lambda x: dict(enumerate([series[x]]))
    
    
    trydf['t6C'] = trydf['Type'].map(i)

    return None



if __name__ == "__main__":
    main()