import numpy as np
from collections import defaultdict
from pylab import *
from math import *
#
#This function loads a file with the name given into to np arrays x and y.
#Parameters: Name of the file to load. Filt : if not -1, filters out only the
#classes in the given set.
#Returns: nparray x with feature information and nparray y with label information, number of different labels k
def load_file(name, filt=-1,uni=False):
    file = open(name,'r')
    x=[]
    y=[]
    for line in file:
        if not line.startswith('#'):
            l=line.split(',')
            xi=[]
            f=len(l)-1
            if uni:
                f=1
            for i in range(0,f):
                xi.append(float(l[i]))
            yi=float(l[-1])
            if not yi == 0 and (filt==-1 or yi in filt):
                y.append(yi)
                x.append(xi)
    return np.array(x), np.array(y)

#
#This function puts together the results of loading all the data
#that is split into different participants data
#Params: filt: set that filters out classes with labels only in the set. -1 as
#default, which means no filtering happens.
#Returns: np araay x with feature information and np array y with label information
def put_files_together(filt=-1,uni=False):
    x,y=load_file("./data/%d.csv"%1,filt,uni)
    for i in range(2,16):
        xi,yi=load_file("./data/%d.csv"%i,filt,uni)
        x=np.concatenate((x,xi))
        y=np.concatenate((y,yi))
    return x,y,np.unique(y)

#
#Estimates the mean vector and covariance matrix for each different class
#in the data
#Parameters: x-feature dara vector or matrix. y-label data vector
def estimate_parameters_GDA(x,y,k,uni=False):
    mean_vec = defaultdict(lambda: np.ndarray(0))
    covar_matrix= defaultdict(lambda: np.ndarray(0))
    prior_class=defaultdict(float)
    for j in k:
        ex_number=0
        sum_x=0
        sum_var=0
        for i in range(len(y)):
            if j==y[i]:
                ex_number+=1
                sum_x+=x[i]

        mean_vec[j]=sum_x/ex_number

        for i in range(len(y)):
            if j==y[i]:
                temp=x[i]-mean_vec[j]
                sum_var+=np.outer(temp,temp)

        covar_matrix[j] = sum_var/ex_number
        prior_class[j] = ex_number/len(x)

    mean_vec=dict(mean_vec)
    covar_matrix=dict(covar_matrix)

    if uni:
        for k in mean_vec:
            mean_vec[k] = mean_vec[k][0]
        for k in covar_matrix:
            covar_matrix[k] = covar_matrix[k][0][0]

    return mean_vec, covar_matrix, dict(prior_class)

#
#Calculates all the membership fuctions for each label j in k
#and returns the j belonging to the class with the highest 
#g value.
def g(x,mean,covar,priori,k,uni=False):
    g_j = defaultdict(float)
    for j in k:
        if uni:
            g_j[j]=-math.log10(covar[j])-(x-mean[j])**2/(2*covar[j]**2)+math.log10(priori[j])
        else:
            g_j[j]=-math.log10(np.linalg.det(covar[j]))-(1/2)*dot(dot((x-mean[j]).T,covar[j]),(x-mean[j]))+math.log10(priori[j])
    return sorted(g_j)[0]

def cross_validation(x,y,k=10):

#
#Computes confusion matrix given
def confusion_matrix():



        




x,y,k=put_files_together((1,2))
mean,covar,priori=(estimate_parameters_GDA(x,y,k))
print (g(x[0],mean,covar,priori,k))


