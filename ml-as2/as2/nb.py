import numpy as np
from collections import defaultdict
from pylab import *
from math import *
from sklearn.cross_validation import KFold
from collections import Counter
import random

#
#This function loads a file with the name given into to np arrays x and y.
#Parameters: Name of the file to load. Filt : if not -1, filters out only the
#classes in the given set.
#Returns: nparray x with feature information and nparray y with label information, number of different labels k
def load_file(name, binary=False):
    file = open(name,'r').readlines()
    random.shuffle(file)
    x=[]
    y=[]
    i=0
    for line in file:
        l=line.split(',')
        xi=[]
        if binary:
            for i in range(0,47):
                xi.append(0 if l[i]=='0' else 1)
        else:
            for i in range(0,47):
                xi.append(int(float(l[i])*random.randint(50,200)))  #randomly chooses an email length between 50 an 200 words.
        yi=float(l[-1])
        y.append(yi)
        x.append(xi)

    return np.array(x), np.array(y)

#
#Estimates the alpha vector and priori for each class
#in the data
#Parameters: x-feature dara vector or matrix. y-label data vector
def estimate_parameters_NB(x,y,e=0.00000000000000001):
    prior_class=defaultdict(float)
    alpha = defaultdict(lambda: np.ndarray(0))
    ev = [e]*len(x[0])#length of features (n)
    for j in range(0,2): #i=1 to m
        ex_number=0
        sum_x=0
        for i in range(len(x)):  #j=1 to m
            if j==y[i]:
                ex_number+=1
                sum_x+=x[i]        #sum_x[p] tiene sumatoria de x_p

        alpha[j]=(sum_x+ev)/(ex_number+2*e)

        prior_class[j] = ex_number/len(x)

    return dict(alpha), dict(prior_class)

def classify_NB(x,alpha,priori):
    g=defaultdict(float)
    for i in range(0,2):
        for j in range(len(x)): #1 to n
            g[i]+=x[j]*math.log(alpha[i][j])+(1-x[j])*math.log(1-alpha[i][j])+math.log(priori[i])

    return 0 if g[0]>g[1] else 1

#
#Computes classifier performance evaluators given the parameters and the featue (x) and
#original label data (y).
#Parameters: x: feature matrix or vector. y: label vector. mean: mean vector.
#covar: covariance matrix. priori: priori class vector. uni: defines if the data
#is univariable or not. False by default.
#Return: confusion matrix, precision, recall, f_measuer and accuracy.
def confusion_matrix(x,y,alpha,priori):
    print("calculating performance evaluators...")
    tp=0
    tn=0
    fp=0
    fn=0
    i=0
    for i in range(0,len(x)):
        g_x=g(x[i],alpha,priori)
        #print("gx: %f yi: %f"%(g_x,y[i]))
        if g_x == 1:
            if y[i]==1:
                tp+=1
            else:
                fp+=1
        else:
            if y[i]==0:
                tn+=1
            else:
                fn+=1

    confusion_matrix = [[tp,fp],[fn,tn]]
    print(confusion_matrix)
    precision = 0 if (tp+fp)==0 else tp/(tp+fp) 
    recall = 0 if (tp+fn)==0 else tp/(tp+fn)
    f_measure = 0 if (precision==0 or recall==0) else 2/((1/precision)+(1/recall))
    accuracy = (tp+tn)/(tp+tn+fp+fn)

    return confusion_matrix, precision, recall, f_measure, accuracy

#
#This function performs crossvalidation to find the average training and testing RSE's of
#given data , given a degree and a k-factor.
#Parameters: x - vector or matrix, y - vector, k - integer, d - degree, integer.
#Returns: training and testing RSE's
def cross_validation(x,y,k,k_f=10,th=1,uni=False):

    traine={'p':0,'r':0,'f_m':0,'a':0}
    teste={'p':0,'r':0,'f_m':0,'a':0}

    kf = KFold(len(x),k_f)
    for train, test in kf:
        mean,covar,priori=(estimate_parameters_GDA(x[train],y[train],k,uni))
        #print(mean,covar,priori)

        rest=confusion_matrix(x[train],y[train],alpha,priori)
        rese=confusion_matrix(x[test],y[test],alpha,priori)

        traine["p"]+=rest[1]
        traine["r"]+=rest[2]
        traine["f_m"]+=rest[3]
        traine["a"]+=rest[4]

        teste["p"]+=rese[1]
        teste["r"]+=rese[2]
        teste["f_m"]+=rese[3]
        teste["a"]+=rese[4]

    traine["p"]/=k_f
    traine["r"]/=k_f
    traine["f_m"]/=k_f
    traine["a"]/=k_f

    teste["p"]/=k_f
    teste["r"]/=k_f
    teste["f_m"]/=k_f
    teste["a"]/=k_f

    return traine, teste





x,y=(load_file("./data/spambase.data.txt",True))
alpha,priori=(estimate_parameters_NB(x,y,0))
#print(alpha,priori)
#n=list(y).index(1)
#print(classify_NB(x[n],alpha,priori))
#print(y[n])