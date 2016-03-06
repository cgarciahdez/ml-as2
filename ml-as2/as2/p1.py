import numpy as np
from collections import defaultdict
from pylab import *
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
def estimate_parameters_multiGDA(x,y,k):
    mean_vec = defaultdict(lambda: np.ndarray(0))
    covar_matrix= defaultdict(lambda: np.ndarray(0))
    for j in k:
        ex_number=0
        sum_x=0
        sum_var=0
        for i in range(len(y)):
            if j==y[i]:
                ex_number+=1
                sum_x+=x[i]

        mean_vec[j]=sum_x/ex_number
        print(mean_vec[j])

        for i in range(len(y)):
            if j==y[i]:
                temp=x[i]-mean_vec[j]
                sum_var+=np.outer(temp,temp)

        covar_matrix[j]=sum_var/ex_number

    return dict(mean_vec), dict(covar_matrix)

#
#Estimates the mean vector and covariance matrix for each different class
#in the data
#Parameters: x-feature dara vector or matrix. y-label data vector
def estimate_parameters_uniGDA(x,y,k):
    mean_vec = defaultdict(lambda: np.ndarray(0))
    covar_matrix= defaultdict(lambda: np.ndarray(0))
    for j in k:
        ex_number=0
        sum_x=0
        sum_var=0
        for i in range(len(y)):
            if j==y[i]:
                ex_number+=1
                sum_x+=x[i]

        mean_vec[j]=sum_x/ex_number
        print(mean_vec[j])

        for i in range(len(y)):
            if j==y[i]:
                temp=x[i]-mean_vec[j]
                sum_var+=np.outer(temp,temp)

        covar_matrix[j]=sum_var/ex_number

    return dict(mean_vec), dict(covar_matrix)


    

x,y,k=put_files_together((1,2),True)
print(estimate_parameters_multiGDA(x,y,k))

