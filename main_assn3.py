# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 21:09:23 2017

@author: sap
"""
#import sys
#sys.path.insert(0, '/Users/sap/Downloads/chandni_ml/assn2')
import main_assn2 as assn2
import os, struct
import matplotlib as plt
from array import array as pyarray
from numpy import array, int8, uint8, zeros as np
import pandas as pd
import scipy.sparse as sparse
import scipy.linalg as linalg
from pylab import *
from numpy import *

#from sklearn.metrics import confusion_matrix as cm

def load_mnist(dataset="training", digits=range(10), path='C:\\Users\\Khushi\\Downloads\\MNIST_data'):
    
    """
    Adapted from: http://cvxopt.org/applications/svm/index.html?highlight=mnist
    """
    if dataset == "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images.idx3')
        fname_lbl = os.path.join(path, 't10k-labels.idx1')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")
   
#    if dataset == "training":
#        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
#        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
#    elif dataset == "testing":
#        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
#        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
   
    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels

def get_mu_zcvpr(X, n, X_mu = None, C = None):
    # Use the first n components of pca to reconstruct
    X = np.asmatrix(X)
    if X_mu is None:
        X_mu = np.mean(X,axis=0)
    Z = X - X_mu # subtracting mean of X from X
    if C is None:
        C = np.cov(Z,rowvar=False) # covariance matrix
        
    [eVals,V] = linalg.eigh(C) # where V is eigenvectors
    V = np.flipud(V.T)
    P = np.dot(Z, V.T)
    R = np.dot(P[:,:n], V[:n,:])  # R is recovered Z
    Rec = R + X_mu # Rec is recovered X
    return X_mu,Z,C,V[:n,:],P[:,:n],R,Rec
    
def get_recognized_label_from_classifier(xf, labels, classifier_func, *args):
    #xf, labels, NF, NM, mu_F,cov_F,mu_M,cov_M):
#    print 'Using classifier: %s' % classifier_func.__name__

    if classifier_func.__name__ == 'bayes_classfier_2d_x_F':
        P_x_F = assn2.bayes_classfier_2d_x_F(xf, *args)
        P_x_M = assn2.bayes_classfier_2d_x_M(xf, *args)
#        print "Bayes[feature1_val, feature2_val]: [%d,%d] = %g" % (xf[0], xf[1], P_x_F)
    elif classifier_func.__name__ == 'hist_classfier_2d_x_F':
        P_x_F = assn2.hist_classfier_2d_x_F(xf, *args)
        P_x_M = assn2.hist_classfier_2d_x_M(xf, *args)
#        print "Hist[feature1_val, feature2_val]: [%d,%d] = %g" % (xf[0], xf[1], P_x_F)
    else:
        print "INVALID Classifier"
        return None
        
    if P_x_F > 0.5:
        recognized_label = labels[0]
    else:
        recognized_label = labels[1]
        
    return recognized_label, P_x_F, P_x_M

def get_accuracy_from_conf_matrix(true_, pred_):
    from sklearn.metrics import confusion_matrix as cm
    cm = cm(true_, pred_)
    TP = cm[0][0]
    TN = cm[1][1]
    FP = cm[1][0]
    FN = cm[0][1]    
    accuracy = float(TP + TN) /  (TP + FN + FP + TN)
    return accuracy
    
def main():
    images, labels = load_mnist('training', digits=[3,7])
    
    # converting from NX28X28 array into NX784 array
    flatimages = list()
    for i in images:
        flatimages.append(i.ravel())
    X = np.asarray(flatimages)
    
    print("Check shape of matrix", X.shape)
    print("Check Mins and Max Values",np.amin(X),np.amax(X))
    print("\nCheck training vector by plotting image \n")
    img_num = 20
    plt.imshow(X[img_num].reshape(28, 28),interpolation='None', cmap=cm.gray)
    show()
    
    # reconstruct images using first few eigenvectors
    n=2
    X_mu,_,C,_,P,R,Rec = get_mu_zcvpr(X,n)
    
    # plot a recovered Z
    fig = plt.figure()
    plt.imshow(R[img_num].reshape(28, 28),interpolation='None', cmap=cm.gray)
    show()

    # plot a recovered image    
    fig = plt.figure()
    plt.imshow(Rec[img_num].reshape(28, 28),interpolation='None', cmap=cm.gray)
    show()
    
    df_P = pd.DataFrame(P[:,:n],columns=['PC1','PC2'])
    df_P['labels'] = labels
    
    # scatterplot
    plt.figure()
    plt.scatter( df_P['PC1'][df_P['labels']==3], df_P['PC2'][df_P['labels']==3], color='magenta', s=0.5 )
    plt.scatter( df_P['PC1'][df_P['labels']==7], df_P['PC2'][df_P['labels']==7], color='y', s=0.5 )
    plt.show()
    
    # means 
    #check stencil interpretation which shows the mixed image of + n - class
    features = ['PC1', 'PC2']
    labels_ = [7, 3]
    label_type = 'labels'
    n_bins = 25
    NF,NM,mu_F,mu_M,cov_F,cov_M,Xh_min, Xh_max,Xhs_min, Xhs_max, \
        bins_h,bins_hs,bins_hm,bins_hsm,H_f,H_m \
        = assn2.create_2d_histogram_classifier(df_P, n_bins, labels_, label_type, features)

    H_f_bayes = assn2.create_bayes_2d_histogram(bins_h,  bins_hs,  mu_F, cov_F, NF)
    H_m_bayes = assn2.create_bayes_2d_histogram(bins_hm, bins_hsm, mu_M, cov_M, NM)
    
    # images(p_) = 7; images(n_) = 3
    p_ = 3 # index of first label 7
    Xp = X[p_]
    _,Zp,_,_,Pp,Rp,Recp = get_mu_zcvpr(Xp, 2, X_mu, C)
    
    
    n_ = 0 # index of first label 3
    Xn = X[n_]
    _,Zn,_,_,Pn,Rn,Recn = get_mu_zcvpr(Xn, 2, X_mu, C)
    
    # 9. Calculate probabilities
    xf = df_P.ix[[p_,n_],['PC1','PC2']].values
    
    for i in range(0,len(xf)):
        # 9.a Using Histogram to calculate Probabilites
        recognized_digit_hist, P_hist_p_, P_hist_n_ = get_recognized_label_from_classifier( \
            xf[i], labels_, \
            assn2.hist_classfier_2d_x_F, n_bins, Xh_min, Xh_max, Xhs_min, Xhs_max, H_f, H_m)
        print "P_Hist_p for [PC1, PC2]: [%d,%d] = %g" % (xf[i][0], xf[i][1], P_hist_p_)
        # 9.b Using Gaussian pdf to calculate Probabilites
        recognized_digit_bayes, P_bayes_p_, P_bayes_n_ = get_recognized_label_from_classifier( \
            xf[i], labels_, \
            assn2.bayes_classfier_2d_x_F, NF, NM, mu_F,cov_F,mu_M,cov_M)
        print "P_Bayes_p for [PC1, PC2]: [%d,%d] = %g" % (xf[i][0], xf[i][1], P_bayes_p_)
        
    # 10. Evaluate Training accuracy
    pc_all = df_P[['PC1','PC2']].values
    for i in range(0,len(pc_all)):
        recognized_digit_bayes, P_bayes_p_, P_bayes_n_ = get_recognized_label_from_classifier( \
            pc_all[i], labels_, \
            assn2.bayes_classfier_2d_x_F, NF, NM, mu_F,cov_F,mu_M,cov_M)
        df_P.ix[i,'recognized_digit_bayes']= recognized_digit_bayes
            
        recognized_digit_hist, P_hist_p_, P_hist_n_ = get_recognized_label_from_classifier( \
            pc_all[i], labels_, \
            assn2.hist_classfier_2d_x_F, n_bins, Xh_min, Xh_max, Xhs_min, Xhs_max, H_f, H_m)
        df_P.ix[i,'recognized_digit_hist'] = recognized_digit_hist
        
    accuracy_hist = get_accuracy_from_conf_matrix(df_P['labels'].values, df_P['recognized_digit_hist'].values)
    print 'Histogram  Classifier accuracy = %f' % accuracy_hist
    
    accuracy_bayes = get_accuracy_from_conf_matrix(df_P['labels'].values, df_P['recognized_digit_bayes'].values)
    print 'Bayes Classifier accuracy = %f' % accuracy_bayes
    
    
    return None    
    
if __name__ == '__main__':
    main()