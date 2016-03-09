import numpy as np
from collections import defaultdict
from pylab import *
from math import *
from sklearn.cross_validation import KFold
from collections import Counter
import random
from matplotlib import pyplot as plt


#
#This function loads a file with the name given into to np arrays x and y.
#Parameters: Name of the file to load. Filt : if not -1, filters out only the
#classes in the given set.
#Returns: nparray x with feature information and nparray y with label information, number of different labels k
def load_file(name, filt=-1,uni=False):
    file = open(name,'r').readlines()
    random.shuffle(file)
    x=[]
    y=[]
    i=0
    for line in file:
        if not line.startswith('#'):
            l=line.split(',')
            xi=[]
            f=len(l)-1
            if uni:
                f=2
            for i in range(1,f):
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
    print("loading data...")
    x,y=load_file("./data/%d.csv"%1,filt,uni)
    for i in range(2,2):
        xi,yi=load_file("./data/%d.csv"%i,filt,uni)
        x=np.concatenate((x,xi))
        y=np.concatenate((y,yi))
    return x[:500],y[:500],np.unique(y[:500])

#
#Estimates the mean vector and covariance matrix for each different class
#in the data
#Parameters: x-feature dara vector or matrix. y-label data vector
def estimate_parameters_GDA(x,y,k,uni=False):
    print("estimating parameters...")
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
        print()
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
            g_j[j]=-math.log(covar[j])-((x-mean[j])**2/(2*(covar[j]**2)))+math.log(priori[j])
        else:
            g_j[j]=-math.log(np.linalg.det(covar[j]))-(1/2)*dot(dot((x-mean[j]).T,covar[j]),(x-mean[j]))+math.log(priori[j])
    return sorted(g_j, key=g_j.get, reverse=True)[0]

#
#Calculates all the membership fuctions for both classes
#and returns the one depending on the threshold
#g value.
def g_2class(x,mean,covar,priori,k,th=1,uni=False):
    g_j = defaultdict(float)
    for j in k:
        if uni:
            g_j[j]=-math.log(covar[j])-(x-mean[j])**2/(2*covar[j]**2)+math.log(priori[j])
        else:
            g_j[j]=-math.log(np.linalg.det(covar[j]))-(1/2)*dot(dot((x-mean[j]).T,covar[j]),(x-mean[j]))+math.log(priori[j])
    #print(dict(g_j))
    if (g_j[k[1]]/g_j[k[0]])>th:
        return k[0]
    else:
        return k[1]

#
#This function performs crossvalidation to find the average training and testing RSE's of
#given data , given a degree and a k-factor.
#Parameters: x - vector or matrix, y - vector, k - integer, d - degree, integer.
#Returns: training and testing RSE's
def cross_validation(x,y,k,k_f=10,th=1,uni=False):
    print("cross validating...")
    #print(len(k))

    if len(k)==2:
        traine={'p':0,'r':0,'f_m':0,'a':0}
        teste={'p':0,'r':0,'f_m':0,'a':0}
    else:
        traine={'p':Counter({}),'r':Counter({}),'f_m':Counter({}),'a':0}
        teste={'p':Counter({}),'r':Counter({}),'f_m':Counter({}),'a':0}
        

    kf = KFold(len(x),k_f)
    for train, test in kf:
        mean,covar,priori=(estimate_parameters_GDA(x[train],y[train],k,uni))
        #print(mean,covar,priori)

        if len(k)==2:
            rest=confusion_matrix_2class(x[train],y[train],k,mean,covar,priori,th,uni)
            rese=confusion_matrix_2class(x[test],y[test],k,mean,covar,priori,th,uni)

            traine["p"]+=rest[1]
            traine["r"]+=rest[2]
            traine["f_m"]+=rest[3]
            traine["a"]+=rest[4]

            teste["p"]+=rese[1]
            teste["r"]+=rese[2]
            teste["f_m"]+=rese[3]
            teste["a"]+=rese[4]
        else:
            rest=confusion_matrix_nclass(x[train],y[train],k,mean,covar,priori,uni)
            rese=confusion_matrix_nclass(x[test],y[test],k,mean,covar,priori,uni)
            #print(rest)
            #print(rese)


            traine["p"]=traine['p']+Counter(rest[1])
            traine["r"]=traine['r']+Counter(rest[2])
            traine["f_m"]=traine['f_m']+Counter(rest[3])
            traine["a"]+=rest[4]

            teste["p"]=teste['p']+Counter(rese[1])
            teste["r"]=teste['r']+Counter(rese[2])
            teste["f_m"]=teste['f_m']+Counter(rese[3])
            teste["a"]+=rese[4]

    if len(k)==2:
        traine["p"]/=k_f
        traine["r"]/=k_f
        traine["f_m"]/=k_f
        traine["a"]/=k_f

        teste["p"]/=k_f
        teste["r"]/=k_f
        teste["f_m"]/=k_f
        teste["a"]/=k_f

    else:
        traine["p"]=dict(traine['p'])
        traine["r"]=dict(traine['r'])
        traine["f_m"]=dict(traine['f_m'])
        traine["a"]/=k_f

        teste["p"]=dict(teste['p'])
        teste["r"]=dict(teste['r'])
        teste["f_m"]=dict(teste['f_m'])
        teste["a"]/=k_f

        for key in traine['p']:
            traine["p"][key]/=k_f
            traine["r"][key]/=k_f
            traine["f_m"][key]/=k_f

        for key in teste['p']:
            teste["p"][key]/=k_f
            teste["r"][key]/=k_f
            teste["f_m"][key]/=k_f

    return traine, teste

#
#Computes classifier performance evaluators given the parameters and the featue (x) and
#original label data (y).
#Parameters: x: feature matrix or vector. y: label vector. mean: mean vector.
#covar: covariance matrix. priori: priori class vector. uni: defines if the data
#is univariable or not. False by default.
#Return: confusion matrix, precision, recall, f_measuer and accuracy.
def confusion_matrix_2class(x,y,k,mean,covar,priori,th=1,uni=False):
    print("calculating performance evaluators...")
    tp=0
    tn=0
    fp=0
    fn=0
    i=0
    for i in range(0,len(x)):
        g_x=g_2class(x[i],mean,covar,priori,k,th,uni)
        #print("gx: %f yi: %f"%(g_x,y[i]))
        if g_x == k[0]:
            if y[i]==k[0]:
                tp+=1
            else:
                fp+=1
        else:
            if y[i]==k[1]:
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
#Computes classifier performance evaluators given the parameters and the featue (x) and
#original label data (y).
#Parameters: x: feature matrix or vector. y: label vector. mean: mean vector.
#covar: covariance matrix. priori: priori class vector. uni: defines if the data
#is univariable or not. False by default.
#Return: confusion matrix, precision, recall, f_measuer and accuracy.
def confusion_matrix_nclass(x,y,k,mean,covar,priori,uni=False):
    print("calculating performance evaluators...")
    n = defaultdict(int)   #Dictionary to assign temporary indexes to each class
    for j in range(0,len(k)):
        n[k[j]]=j
    confusion_matrix = np.zeros((len(k),len(k)), dtype=float)
    
    for i in range(0,len(x)):
        g_x=g(x[i],mean,covar,priori,k,uni)
        confusion_matrix[n[g_x]][n[y[i]]]+=1

    precision = defaultdict(float)
    recall = defaultdict(float)
    f_measure = defaultdict(float)
    accuracy = defaultdict(float)

    total = confusion_matrix.sum()
    row_sum = confusion_matrix.sum(axis=1)
    column_sum = confusion_matrix.sum(axis=0)
    accuracy = 0
    for j in range(0,len(k)):
        accuracy+=confusion_matrix[j][j]
    accuracy/=total
    for j in k:
        precision[j]=confusion_matrix[n[j]][n[j]]/row_sum[n[j]]
        recall[j]=confusion_matrix[n[j]][n[j]]/column_sum[n[j]]
        f_measure[j] = 2* (precision[j]*recall[j])/(precision[j]+recall[j])


    return confusion_matrix, dict(precision), dict(recall), dict(f_measure), accuracy


#x,y,k=load_iris("./data/iris.data.txt",two=True)
x,y,k=put_files_together((5,6,7))
print(x.shape)
if(len(k)==2):
    precision=[]
    recall=[]
    for i in range(1,10):
        p=cross_validation(x,y,k,th=i/10)
        precision.append(p[1]['p'])
        recall.append(p[1]['r'])
    print (precision)
    print(recall)
    plt.plot(recall,precision)
    plt.show()
else:
    print (cross_validation(x,y,k))


#print (cross_validation(x,y,k,th=1))


