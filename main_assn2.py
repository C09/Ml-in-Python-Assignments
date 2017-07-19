# -*- coding: utf-8 -*-

"""

Created on Tue Jun 06 20:03:23 2017



@author: Chandni J Purohit

"""

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from numpy import *

import math



def map_x_to_bin_id( X, n_bins, X_min, X_max):

    bin_ids = np.round(((n_bins-1)*(X-X_min)/(X_max-X_min))).astype('int32')

    return  bin_ids





def Histogram1DClassifier(X,n_bins,X_min,X_max):

    #bin_ids = np.round(((n_bins-1)*(X-X_min)/(X_max-X_min))).astype('int32');

    bin_ids = map_x_to_bin_id(X, n_bins, X_min, X_max)

    H = np.zeros(n_bins).astype('int32');

    for i,bin_id in enumerate(bin_ids):

        H[bin_id] += 1

    

    bw = float(X_max - X_min) / n_bins

    X_bins = X_min + bw *np.array(range(1,n_bins+1))

    return H, X_bins, bin_ids



def Histogram2DClassifier(X1,X2,n_bins1,n_bins2,df,features):

    X1_min = np.min(df[features[0]])

    X1_max = np.max(df[features[0]])

    X2_min = np.min(df[features[1]])

    X2_max = np.max(df[features[1]])

    

    H1,bins1,bin_ids1=Histogram1DClassifier(X1,n_bins1,X1_min,X1_max)

    H2,bins2,bin_ids2=Histogram1DClassifier(X2,n_bins2,X2_min,X2_max)

    H2d =  np.zeros((n_bins1,n_bins2))

    for x1_,x2_ in zip(bin_ids1,bin_ids2):

        H2d[x1_,x2_]+=1

           

    return H2d,bins1,bins2   

  

def plot_3dhist(data_array,bins_h,bins_hs):



    fig=plt.figure()

    ax=fig.add_subplot(111,projection='3d')

    x_data,y_data = np.meshgrid(np.arange(data_array.shape[1]),np.arange(data_array.shape[0]))

    x_data = x_data.flatten()

    y_data = y_data.flatten()

    z_data =  data_array.flatten()

    ax.bar3d(x_data,

             y_data,

             np.zeros(len(z_data)),

             1,1,z_data)

    ax.w_xaxis.set_ticklabels(bins_h)

    ax.w_yaxis.set_ticklabels(bins_hs)    

    plt.show()

    return None





def norm_pdf_multivariate(x, mu, sigma):

    size = len(x)

    if size == len(mu) and (size, size) == sigma.shape:

        det = linalg.det(sigma)

        if det == 0:

            raise NameError("The covariance matrix can't be singular")



        norm_const = 1.0/ ( math.pow((2*pi),float(size)/2) * math.pow(det,1.0/2) )

        x_mu = matrix(x - mu)

        inv = sigma.I        

        result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))

        return norm_const * result

    else:

        raise NameError("The dimensions of the input don't match")



def bayes_classfier_2d_x_F(x, NF, NM, mu_F,cov_F,mu_M,cov_M):

    return NF*norm_pdf_multivariate(x,mu_F,cov_F) / (NF*norm_pdf_multivariate(x,mu_F,cov_F) + NM*norm_pdf_multivariate(x,mu_M,cov_M))

    

def bayes_classfier_2d_x_M(x, NF, NM, mu_F,cov_F,mu_M,cov_M):

    return NM*norm_pdf_multivariate(x,mu_M,cov_M) / (NF*norm_pdf_multivariate(x,mu_F,cov_F) + NM*norm_pdf_multivariate(x,mu_M,cov_M))

    

def hist_classfier_2d_x_F(x, n_bins, Xh_min, Xh_max, Xhs_min, Xhs_max, H_f, H_m):

    bin_id_h_  = map_x_to_bin_id(x[0], n_bins, Xh_min, Xh_max)

    bin_id_hs_ = map_x_to_bin_id(x[1], n_bins, Xhs_min, Xhs_max)

    return (H_f[bin_id_h_,bin_id_hs_] / (H_f[bin_id_h_,bin_id_hs_] + H_m[bin_id_h_,bin_id_hs_]))

    

def hist_classfier_2d_x_M(x, n_bins, Xh_min, Xh_max, Xhs_min, Xhs_max, H_f, H_m):

    bin_id_h_  = map_x_to_bin_id(x[0], n_bins, Xh_min, Xh_max)

    bin_id_hs_ = map_x_to_bin_id(x[1], n_bins, Xhs_min, Xhs_max)

    return (H_m[bin_id_h_,bin_id_hs_] / (H_f[bin_id_h_,bin_id_hs_] + H_m[bin_id_h_,bin_id_hs_]))

        

def create_bayes_2d_histogram(bins_h, bins_hs, mu_, cov_, N):

    H_f_bayes =  np.zeros((len(bins_h), len(bins_hs)))

    width_h_ = bins_h[1] - bins_h[0]

    width_hs_ = bins_hs[1] - bins_hs[0]

    for i_h_, h_ in enumerate(bins_h):

        for i_hs_, hs_ in enumerate(bins_hs):

            pbt_ = width_h_ * width_hs_ * norm_pdf_multivariate([h_,hs_], mu_, cov_)

           

            H_f_bayes[i_h_,i_hs_] = N * pbt_



    plot_3dhist(H_f_bayes, bins_h, bins_hs) 

    return H_f_bayes



def create_2d_histogram_classifier(df, n_bins, labels, label_type, features):

    # create histogram for females for height

    X_h_f = df[df[label_type]==labels[0]][features[0]].values

    X_hs_f =df[df[label_type]==labels[0]][features[1]].values

    

    X_h_m= df[df[label_type]==labels[1]][features[0]].values

    X_hs_m =df[df[label_type]==labels[1]][features[1]].values  

    

    n_bins1 = n_bins

    n_bins2 = n_bins1

    Xh_min = np.min(df[features[0]])

    Xh_max = np.max(df[features[0]])

    Xhs_min = np.min(df[features[1]])

    Xhs_max = np.max(df[features[1]])

    

    # 2. Construct separate 2D histograms for male and female heights.

    H_f, bins_h,bins_hs   = Histogram2DClassifier(X_h_f, X_hs_f, n_bins1, n_bins2, df, features)

    H_m, bins_hm,bins_hsm = Histogram2DClassifier(X_h_m, X_hs_m, n_bins1, n_bins2, df, features)

    plot_3dhist(H_f, bins_h,  bins_hs ) 

    plot_3dhist(H_m, bins_hm, bins_hsm) 

   

    df_F=df[df[label_type]==labels[0]]

    df_M=df[df[label_type]==labels[1]]

    NF = len(df_F) # number of females

    NM = len(df_M) # number of males

    

    # 3. Find the parameters of two 2D Gaussian models for the 2 PDFs to describe the data

    mu_F = np.array(df_F[features].mean(axis = 0))

    mu_M = np.array(df_M[features].mean(axis = 0))

    cov_F = np.asmatrix(df_F[features].cov().values)

    cov_M = np.asmatrix(df_M[features].cov().values)

    return NF,NM,mu_F,mu_M,cov_F,cov_M,Xh_min, Xh_max,Xhs_min, Xhs_max,bins_h,bins_hs,bins_hm,bins_hsm,H_f,H_m

    

def main():

    excelfile=r"C:\Users\Khushi\Downloads\Assignment_2_Data_and_Template.xlsx";

    

    # read from Excel into dataframe

    df = pd.read_excel(excelfile, sheetname='Data')

    n_bins = 8

    

    features = ['Height', 'HandSpan']

    labels = ['Female', 'Male']

    label_type = 'Sex'

    

    NF,NM,mu_F,mu_M,cov_F,cov_M,Xh_min, Xh_max,Xhs_min, Xhs_max,bins_h,bins_hs,bins_hm,bins_hsm,H_f,H_m = create_2d_histogram_classifier(df, n_bins, labels, label_type, features)

    

    # 4. Based on the histograms and Gaussian models, compute the likely gender (given as 

    #    the probability of being female) of individuals with measurements as given below 

    #    (Height in inches, handspan in centimeters)

    xf = np.array([[69,17.5],[66,22],[70,21.5],[69,23.5]])

    

    # 4.a Using Histogram classifier to calculate Probabilites

    for i in range(0,len(xf)):

        bin_id_h_ = map_x_to_bin_id(xf[i][0], n_bins, Xh_min, Xh_max)

        bin_id_hs_ = map_x_to_bin_id(xf[i][1], n_bins, Xhs_min, Xhs_max)

        print "Hist [%d,%d] : %g" % (xf[i][0], xf[i][1], (H_f[bin_id_h_,bin_id_hs_] / (H_f[bin_id_h_,bin_id_hs_] + H_m[bin_id_h_,bin_id_hs_])))



    # 4.b Using Gaussian pdf classifier to calculate Probabilites

    for i in range(0,len(xf)):

        P_x_F = bayes_classfier_2d_x_F(xf[i], NF, NM, mu_F,cov_F,mu_M,cov_M)

        P_x_M = bayes_classfier_2d_x_M(xf[i], NF, NM, mu_F,cov_F,mu_M,cov_M)

        print "Bayes [%d,%d] : %g" % (xf[i][0], xf[i][1], P_x_F)

    

    # create multivariate pdf

    H_f_bayes = create_bayes_2d_histogram(bins_h,  bins_hs,  mu_F, cov_F, NF)

    H_m_bayes = create_bayes_2d_histogram(bins_hm, bins_hsm, mu_M, cov_M, NM)



    print 'Approx histogram sum - Females : %.02f (expected: %.02f)' % (H_f_bayes.sum(), NF)

    print 'Approx histogram sum - Males   : %.02f (expected: %.02f)' % (H_m_bayes.sum(), NM)

    print "Done: exiting main()"

    return None

    

    

if __name__ == "__main__":

    main()