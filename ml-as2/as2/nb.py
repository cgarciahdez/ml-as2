import numpy as np
from collections import defaultdict
from pylab import *
from math import *
from sklearn.cross_validation import KFold
from collections import Counter
import random

#
#This function loads a file with the name given into to np arrays x and y.
#Parameters: Name of the file to load. Binary : if True, converts features to binary (0 or not 0)
#Returns: nparray x with feature information and nparray y with label information
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
            n = random.randint(50,100) #randomly chooses an email length between 150 an 200 words.
            for i in range(0,47):
                xi.append(int(math.ceil(float(l[i])/100*n)))  
            xi.append(n)
        yi=float(l[-1])
        y.append(yi)
        x.append(xi)

    return np.array(x), np.array(y)

#
#Estimates the alpha vector and priori for each class in the data (0 and 1)
#Parameters: x-feature dara vector or matrix. y-label data vector. e: smoothing factor. 
#bino: True if features are binomial. If False, it is assumed they are binary 
#Returns: model params alpha and priori class
def estimate_parameters_NB(x,y,e=0.00000000000000001,bino=False):
    prior_class=defaultdict(float)
    alpha = defaultdict(lambda: np.ndarray(0))
    ev = [e]*len(x[0])#length of features (n)
    for j in range(0,2): #i=1 to m
        ex_number=0
        ex_bin=0
        sum_x=0
        for i in range(len(x)):  #j=1 to m
            if j==y[i]:
                ex_number+=1
                sum_x+=x[i]        #sum_x[p] tiene sumatoria de x_p
                if bino:
                    ex_bin+=x[i][-1]   #p_i

        if bino:
            alpha[j]=(sum_x+ev)/(ex_bin+2*e)
        else:
            alpha[j]=(sum_x+ev)/(ex_number+2*e)

        prior_class[j] = ex_number/len(x)

    return dict(alpha), dict(prior_class)

#
#This function classifys the data between the cñasse 0 and 1 by calculating
#the membership for each of them and returning the highest one
#Params: x: feature vector to be classified. alpha: alpha of model. priori: priori clasees of model
#bino: True if features are binomial. If False, it is assumed they are binary 
#Return: class where x was classified
def classify_NB(x,alpha,priori,bino=False):
    g=defaultdict(float)
    l = len(x)-1 if bino else len(x)
    for i in range(0,2):
        for j in range(l): #1 to n
            if not bino:
                g[i]+=x[j]*math.log(alpha[i][j])+(1-x[j])*math.log(1-alpha[i][j])+math.log(priori[i])
            else:
                #print("p: %f y x_j: %f"%(x[-1],x[j]))
                #print(nCr(x[-1],x[j]))
                g[i]+=math.log(nCr(x[-1],x[j])*(alpha[i][j]**x[j])*((1-alpha[i][j])**(x[-1]-x[j])))+math.log(priori[i])

    return 0 if g[0]>g[1] else 1

#
#This auxiliary function calculates n choose r
#Params: integers n and r
#Returns: n choose r
def nCr(n,r):
    if r>n:
        return 0
    f = math.factorial
    return f(n) // f(r) // f(n-r)

#
#Computes classifier performance evaluators given the parameters and the featue (x) and
#original label data (y).
#Parameters: x: feature matrix or vector. y: label vector. alpha: alpha vector
#priori: priori class vector.
#bino: True if features are binomial. If False, it is assumed they are binary 
#Return: confusion matrix, precision, recall, f_measure and accuracy.
def confusion_matrix(x,y,alpha,priori,bino=False):
    tp=0
    tn=0
    fp=0
    fn=0
    i=0
    l = len(x)-1 if bino else len(x)
    for i in range(0,l):
        g_x=classify_NB(x[i],alpha,priori,bino)
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
    #print(confusion_matrix)
    precision = 0 if (tp+fp)==0 else tp/(tp+fp) 
    recall = 0 if (tp+fn)==0 else tp/(tp+fn)
    f_measure = 0 if (precision==0 or recall==0) else 2/((1/precision)+(1/recall))
    accuracy = (tp+tn)/(tp+tn+fp+fn)

    return confusion_matrix, precision, recall, f_measure, accuracy

#
#This function performs crossvalidation to find the average performance
#evaluators of the NB classifier
#Parameters: x - feature matrix, y - label vector, k - number of folds. e: smoothing factor
#bino: True if features are binomial. If False, it is assumed they are binary 
#Returns: training and testing RSE's
def cross_validation(x,y,k_f=10,e=0.00000000000000001,bino=False):

    traine={'p':0,'r':0,'f_m':0,'a':0}
    teste={'p':0,'r':0,'f_m':0,'a':0}

    kf = KFold(len(x),k_f)
    for train, test in kf:
        alpha,priori=(estimate_parameters_NB(x[train],y[train],e,bino))
        #print(mean,covar,priori)

        rest=confusion_matrix(x[train],y[train],alpha,priori,bino)
        rese=confusion_matrix(x[test],y[test],alpha,priori,bino)

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





x,y=(load_file("./data/spambase.data.txt",binary=True))
cv = cross_validation(x,y,e=1,bino=False)
print("Running NB with bernoulli features")
print("Assuming spam to be Positive and non spam to me Negative:\n\n")
print ("The training performance evaluators are:\n"\
"Precision: %f\n Recall: %f\n F_measure: %f\n Accuracy: %f\n\n"\
"The tesing performance evaluators are:\n"\
"Precision: %f\n Recall: %f\n F_measure: %f\n Accuracy: %f\n\n" % (cv[0]['p'],cv[0]['r'],cv[0]['f_m'],cv[0]['a'],cv[1]['p'],cv[1]['r'],cv[1]['f_m'],cv[1]['a']))

x,y=(load_file("./data/spambase.data.txt",binary=False))
cv = cross_validation(x,y,e=1,bino=True)
print("Running NB with Binomial features")
print("Assuming spam to be Positive and non spam to me Negative:\n\n")
print ("The training performance evaluators are:\n"\
"Precision: %f\n Recall: %f\n F_measure: %f\n Accuracy: %f\n\n"\
"The tesing performance evaluators are:\n"\
"Precision: %f\n Recall: %f\n F_measure: %f\n Accuracy: %f\n\n" % (cv[0]['p'],cv[0]['r'],cv[0]['f_m'],cv[0]['a'],cv[1]['p'],cv[1]['r'],cv[1]['f_m'],cv[1]['a']))

