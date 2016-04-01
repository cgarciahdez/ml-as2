from sklearn.datasets import fetch_mldata
import numpy as np
import os
from pylab import *
import pdb
import random
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import shuffle
from sklearn.cross_validation import KFold
from collections import defaultdict
from collections import Counter

#
#This function loads MNIST into to np arrays x and y.
#Params: two class true if only two classes d degree (for mapping)
#Returns: nparray x with feature information and nparray y with label information
def load_data(two_class=False,d=1):
	poly = PolynomialFeatures(degree=d)
	mnist = fetch_mldata('MNIST original')
	mnist.data.shape
	mnist.target.shape
	np.unique(mnist.target)

	X, y = mnist.data / 255., mnist.target
	if two_class:
		ind = [ k for k in range(len(y)) if (y[k]==0 or y[k]==1) ]
		X=X[ind,:]
		y=y[ind]
	y=[int(p) for p in y]
	X=poly.fit_transform(X)

	return np.array(X),np.array(y)

#
#This function loads iris dataset into to np arrays x and y.
#Params: two class true if only two classes d degree (for mapping)
#Returns: nparray x with feature information and nparray y with label information
def load_iris(two_class=False,d=1):
	poly = PolynomialFeatures(degree=d)
	file = open("./data/iris.data.txt",'r').readlines()
	random.shuffle(file)
	labels={'Iris-setosa\n':0,'Iris-versicolor\n':1,'Iris-virginica\n':2}
	x=[]
	y=[]
	i=0
	for line in file:
	    l=line.split(',')
	    if(len(l)>1):
		    xi=[]
		    for i in range(1,len(l)-1):
		        xi.append(float(l[i]))
		    yi=labels[l[-1]]
		    if two_class and yi<2:
		        y.append(yi)
		        x.append(xi)
		    elif not two_class:
		    	y.append(yi)
		    	x.append(xi)

	x=poly.fit_transform(x)
	return np.array(x), np.array(y)

#This function computes the parameters for the logistic regression with two classes
#X y data
#Returns theta vector
def gradient_descent_two_class(X,y):	
	theta = [0.01]*len(X[0])
	diff=10
	step=0.0005
	while(diff>1e-3):
		summ=0
		for i in range(0,len(y)):
			summ+=np.dot(h(X[i],theta)-y[i],X[i])
		theta_o = theta
		theta = theta - step*summ
		diff=np.average(np.absolute(theta_o-theta))
		# to=np.average(theta_o)
		# t=np.average(theta)
		# if t<to:
		# 	step=step*1.05
		# else:
		# 	theta=theta_o
		# 	step=step*0.5
	return theta

#This function computes the parameters for the logistic regression with k classes
#X y data
#Returns theta vector
def gradient_descent_k_class(X,y):
	k=set(y)
	
	diff=10
	step=0.0005
	theta_i = np.array([[0.01]*len(X[0])]*len(k))
	while(diff>1e-3):
		summ=defaultdict(float)
		for i in range(0,len(y)):
			h_ = h_k(X[i],theta_i)
			for j in k:
				y_1=0
				if(j==y[i]):
					y_1=1
				summ[j]+=np.dot(h_[j]-y_1,X[i])
			
		theta_o = np.copy(theta_i)
		for j in range(0,len(theta_i)):
			theta_i[j] = theta_i[j] - step*summ[j]
		diff=np.average(np.absolute(theta_o-theta_i))
	return theta_i

#
#Params: feature vector x, theta vector
#Computes the sigmoid function
#Returns: vector of h values for each class
def h_k(x,theta):
	exps = [] 
	for j in range(0,len(theta)):
		ex = np.dot(np.transpose(theta[j]),x)
		exps.append(np.exp(ex))
	summ = sum(exps)
	return [x / summ for x in exps]

#
#Params: feature vector x, theta vector
#Computes the softmax function
#Returns: h value
def h(x,theta):
	h = - np.dot(np.transpose(theta),x)
	h = np.exp(h)
	h= 1/(1+h)
	return h

#
#Classifies an x in one of two classes
def classify(x,theta):
	h_ = h(x,theta)
	if h_<0.5:
		return 0
	else:
		return 1

#Theta is a vector
#Classifies an x example in one of k classes
def classify_k(x,theta):
	y_hat=defaultdict(float)
	for j in range(len(theta)):
		y_hat[j] = np.dot(np.transpose(theta[j]),x)

	s= sorted(y_hat.items(), key=lambda x: x[1], reverse=True)
	return s[0][0]

#
#Computes classifier performance evaluators given the parameters and the featue (x) and
#original label data (y).
#Parameters: x: feature matrix or vector. y: label vector. mean: mean vector.
#covar: covariance matrix. priori: priori class vector. uni: defines if the data
#is univariable or not. False by default.
#Return: confusion matrix, precision, recall, f_measuer and accuracy.
def confusion_matrix_2(X,y,theta):
    tp=0
    tn=0
    fp=0
    fn=0
    i=0
    for i in range(0,len(X)):
        h_=classify(X[i],theta)
        if h_ == y[i]:
            if y[i]==0:
                tn+=1
            else:
                tp+=1
        else:
            if y[i]==0:
                fp+=1
            else:
                fn+=1

    confusion_matrix = [[tp,fp],[fn,tn]]
    precision = 0 if (tp+fp)==0 else tp/(tp+fp) 
    recall = 0 if (tp+fn)==0 else tp/(tp+fn)
    f_measure = 0 if (precision==0 or recall==0) else 2/((1/precision)+(1/recall))
    accuracy = (tp+tn)/(tp+tn+fp+fn)

    return confusion_matrix, precision, recall, f_measure, accuracy

#
#Computes classifier performance evaluators given the parameters and the featue (x) and
#original label data (y).
#Parameters: x: feature matrix or vector. y: label vector. theta: theta vector.
#Return: confusion matrix, precision, recall, f_measuer and accuracy.
def confusion_matrix_k(X,y,theta,k):
	confusion_matrix = np.zeros((len(k),len(k)), dtype=float)

	for i in range(0,len(X)):
		y_hat = classify_k(X[i],theta)
		#pdb.set_trace()
		confusion_matrix[y_hat][y[i]]+=1

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
	    precision[j]=confusion_matrix[j][j]/row_sum[j]
	    recall[j]=confusion_matrix[j][j]/column_sum[j]
	    f_measure[j] = 2* (precision[j]*recall[j])/(precision[j]+recall[j])


	return confusion_matrix, dict(precision), dict(recall), dict(f_measure), accuracy

#
#This function performs crossvalidation to find the average training and testing evaluators
#given data , given a degree and a k-factor.
#Parameters: x - vector or matrix, y - vector
#Returns: accuracies
def cross_validation(X,y,k_f=10):
	k=set(y)
	if len(k)==2:
	    traine={'p':0,'r':0,'f_m':0,'a':0}
	    teste={'p':0,'r':0,'f_m':0,'a':0}
	else:
	    traine={'p':Counter({}),'r':Counter({}),'f_m':Counter({}),'a':0}
	    teste={'p':Counter({}),'r':Counter({}),'f_m':Counter({}),'a':0}
	    

	kf = KFold(len(X),k_f)
	for train, test in kf:

		if len(k)==2:
			theta = (gradient_descent_two_class(X[train],y[train]))
			rest=confusion_matrix_2(X[train],y[train],theta)
			rese=confusion_matrix_2(X[test],y[test],theta)
			
			traine["p"]+=rest[1]
			traine["r"]+=rest[2]
			traine["f_m"]+=rest[3]
			traine["a"]+=rest[4]

			teste["p"]+=rese[1]
			teste["r"]+=rese[2]
			teste["f_m"]+=rese[3]
			teste["a"]+=rese[4]
		else:
			theta = (gradient_descent_k_class(X[train],y[train]))
			rest=confusion_matrix_k(X[train],y[train],theta,k)
			rese=confusion_matrix_k(X[test],y[test],theta,k)

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


def print_result(two_class=False,d=1,iris=True):
	if iris:
		X,y=load_iris(two_class=two_class,d=d)
	else:
		X,y=load_data(two_class=two_class,d=d)
		X, y = shuffle(X, y, random_state=0)
		X=X[:100,:5]
		y=y[:100]
	k=set(y)
	if(len(k)==2):
		p=cross_validation(X,y)
		print ("The training performance evaluators are:\n"\
		"Precision: %f\n Recall: %f\n F_measure: %f\n Accuracy: %f\n\n"\
		"The tesing performance evaluators are:\n"\
		"Precision: %f\n Recall: %f\n F_measure: %f\n Accuracy: %f\n\n" % (p[0]['p'],p[0]['r'],p[0]['f_m'],p[0]['a'],p[1]['p'],p[1]['r'],p[1]['f_m'],p[1]['a']))
	else:
		p=cross_validation(X,y)
		for j in k:
			print("Results for class %d"%j)
			print ("The training performance evaluators are:\n"\
			"Precision: %f\n Recall: %f\n F_measure: %f\n\n"\
			"The tesing performance evaluators are:\n"\
			"Precision: %f\n Recall: %f\n F_measure: %f\n\n" % (p[0]['p'][j],p[0]['r'][j],p[0]['f_m'][j],p[1]['p'][j],p[1]['r'][j],p[1]['f_m'][j]))
		print ("Training Accuracy: %f\nTesting Accuracy: %f"%(p[0]['a'],p[1]['a']))

print("\n\nTwo class logistic regression:")
print_result(True)
print("\n\nTwo class logistic regression with second degre")
print_result(True,2)
print("\n\nK class logistic regression")
print_result(False)







