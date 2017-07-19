# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 13:51:11 2017

@author: Chandni J Purohit
"""
import pandas as pd
import numpy as np
from scipy import linalg as lg
import itertools
from sklearn.metrics import confusion_matrix as cm
from sklearn.preprocessing import OneHotEncoder

excelfilename = r"C:\Users\Khushi\Downloads\Assignment_4_Data_and_Template.xlsx";
Data = pd.read_excel(excelfilename,sheetname="Training Data")

#X = Data [[17]]
orig=Data['Type']
x6c = orig.reshape(-1,1)


enc = OneHotEncoder(sparse=False)
enc.fit(x6c)
encoded = enc.transform(x6c)

encoded[np.where(encoded == 0)] = -1
                
decoded = encoded.dot(enc.active_features_).astype(int)
#assert np.allclose(x6c, decoded)

decode_enc_1 = encoded.dot(enc.active_features_).astype(int)
