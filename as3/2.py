from sklearn.datasets import fetch_mldata
import numpy as np
import os
from math import *
import pdb
import random
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import shuffle
from sklearn.cross_validation import KFold
from collections import defaultdict
from collections import Counter
from sklearn.neural_network import MLPClassifier

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
	ind = [ k for k in range(len(y)) if (y[k]==0 or y[k]==1 or y[k]==2) ]
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

#
#This function computes the parameters for the logistic regression with k classes
#X y data
#Returns v and w vectors
def gradient_descent_k_class(X,y,q_z,beta=0.5,l_r=0.00001):
	k=set(y)
	
	diff=10
	step=l_r
	#v_i = np.array([[0.2]*q_z]*len(k))
	#w_i = np.array([[0.2]*len(X[0])]*(q_z))
	v_i = np.random.uniform(0.0,0.2,(len(k),q_z))
	w_i = np.random.uniform(0.0,0.2,(q_z,len(X[0])))
	w_m = np.copy(w_i)
	w_o = np.copy(w_i)

	obj_i=0

	for i in range(0,len(y)):
		z=[]	
		for l in range(q_z):
			z.append(h(X[i],w_i[l]))
		z = np.array(z)
		y_hat_i = h_k(z,v_i)
		for j in k:
			y_1=1 if y[i]==j else 0
			obj_i+=y_1*log(y_hat_i[j])

	obj_i=-obj_i

	it=0
	while(diff>1e-3 and it<200):
		it+=1
		summ_v=defaultdict(float)
		summ_w=defaultdict(float)
		
		for i in range(0,len(y)):
			z=[]
			for l in range(q_z):
				z.append(h(X[i],w_i[l]))
			z = np.array(z)
			y_hat = h_k(z,v_i) #this is a list of y's
			for j in k:
				summ_v[j]+=np.dot(y_hat[j]-y[i],z)

			for j in range(q_z):
				sumi=0
				for l in k:
					y_j = 1 if y[i]==l else 0
					sumi+=np.dot(y_hat[l]-y_j,v_i[l][j])
				summ_w[j]+=np.dot(np.dot(sumi,z[j]),np.dot((1-z[j]),X[i]))

		w_m = np.copy(w_o)
		v_o = np.copy(v_i)
		w_o = np.copy(w_i)
		for j in range(0,len(v_i)):
			v_i[j] = v_i[j] - step*summ_v[j]
		for j in range(0,len(w_i)):
			w_i[j] = w_i[j] - step*summ_w[j] + beta*(w_i[j]-w_m[j])

		obj_o=obj_i
		obj_i=0

		z_i=[]

		for i in range(0,len(y)):
			z=[]	
			for l in range(q_z):
				z.append(h(X[i],w_i[l]))
			z = np.array(z)
			y_hat_i = h_k(z,v_i)
			for j in k:
				y_1=1 if y[i]==j else 0
				obj_i+=y_1*log(y_hat_i[j])
			z_i.append(y_hat_i)

		#print(v_i)

		obj_i=-obj_i


		diff=(np.absolute(obj_o-obj_i))

	return v_i, w_i

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
#Classifies an x example in one of k classes
def classify(x,w,v):
	q_z=len(w)
	z=[]	
	for l in range(q_z):
		z.append(h(x,w[l]))
	z = np.array(z)
	y_hat_i = h_k(z,v)
	maxx=y_hat_i[0]
	best=0
	for j in range(len(y_hat_i)):
		if y_hat_i[j]>maxx:
			maxx = y_hat_i[j]
			best=j
	return best

#
#Computes classifier performance evaluators given the parameters and the featue (x) and
#original label data (y).
#Parameters: x: feature matrix or vector. y: label vector. theta: theta vector.
#Return: confusion matrix, precision, recall, f_measuer and accuracy.
def confusion_matrix_k(X,y,w,v,k):
	confusion_matrix = np.zeros((len(k),len(k)), dtype=float)
	for i in range(0,len(X)):
		y_hat = classify(X[i],w,v)
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


	return confusion_matrix, (precision), (recall), (f_measure), accuracy

#
#This function performs crossvalidation to find the average training and testing evaluators
#given data , given a degree and a k-factor.
#Parameters: x - vector or matrix, y - vector
#Returns: accuracies
def cross_validation(X,y,q_z,beta=0.5,l_r=0.0001,k_f=10):
	k=set(y)
	traine={'p':Counter({}),'r':Counter({}),'f_m':Counter({}),'a':0}
	teste={'p':Counter({}),'r':Counter({}),'f_m':Counter({}),'a':0}
	    

	kf = KFold(len(X),k_f)
	for train, test in kf:
		v,w = (gradient_descent_k_class(X[train],y[train],q_z,beta=beta,l_r=l_r))
		rest=confusion_matrix_k(X[train],y[train],w,v,k)
		rese=confusion_matrix_k(X[test],y[test],w,v,k)

		traine["p"]=traine['p']+Counter(rest[1])
		traine["r"]=traine['r']+Counter(rest[2])
		traine["f_m"]=traine['f_m']+Counter(rest[3])
		traine["a"]+=rest[4]

		teste["p"]=teste['p']+Counter(rese[1])
		teste["r"]=teste['r']+Counter(rese[2])
		teste["f_m"]=teste['f_m']+Counter(rese[3])
		teste["a"]+=rese[4]

	traine["p"]=(traine['p'])
	traine["r"]=(traine['r'])
	traine["f_m"]=(traine['f_m'])
	traine["a"]/=k_f

	teste["p"]=(teste['p'])
	teste["r"]=(teste['r'])
	teste["f_m"]=(teste['f_m'])
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
#PErforms cross validation for the scikit method.
#Params: same as other one
#Return: testing ans training accuracy
def scikit_method(X,y,q_z,l_r,beta=0.9,k_f=10):
	
	train_acc=0
	test_acc=0

	clf = MLPClassifier(hidden_layer_sizes=(q_z,),activation='logistic',algorithm='sgd'
		,learning_rate_init=l_r,momentum=beta)
	  

	kf = KFold(len(X),k_f)
	for train, test in kf:
		clf.fit(X[train],y[train])
		train_acc = clf.score(X[train],y[train])
		test_acc = clf.score(X[test],y[test])

	print("Train accuracy for scikit method: %f"%train_acc)
	print("Test accuracy for scikit method: %f"%test_acc)


	return train_acc/k_f, test_acc/k_f



def print_result(q_z,l_r=0.0001,two_class=False,d=1,iris=True):
	if iris:
		X,y=load_iris(two_class=two_class,d=d)
	else:
		X,y=load_data(two_class=two_class,d=d)
		X, y = shuffle(X, y, random_state=0)
		X=X[:200,:10]
		y=y[:200]
	k=set(y)
	if(len(k)==2):
		p=cross_validation(X,y,q_z,beta=0.5,l_r=l_r)
		print ("The training performance evaluators are:\n"\
		"Precision: %f\n Recall: %f\n F_measure: %f\n Accuracy: %f\n\n"\
		"The tesing performance evaluators are:\n"\
		"Precision: %f\n Recall: %f\n F_measure: %f\n Accuracy: %f\n\n" % (p[0]['p'],p[0]['r'],p[0]['f_m'],p[0]['a'],p[1]['p'],p[1]['r'],p[1]['f_m'],p[1]['a']))
	else:
		p=cross_validation(X,y,q_z,beta=0.5,l_r=l_r)
		print ("Manual Training Accuracy: %f\nManual Testing Accuracy: %f"%(p[0]['a'],p[1]['a']))



X,y=load_data()
X, y = shuffle(X, y, random_state=0)
X=X[:200,:10]
y=y[:200]


q_z=int((len(X[0])+len(set(y)))/2)
print("\nWith %d cells in hidden layer"%q_z)
print("Manual:")
print_result(q_z,l_r=0.0001,iris=False)
print("Scikit:")
scikit_method(X,y,q_z,l_r=0.0001,beta=0.5,k_f=10)


q_z=int((len(X[0])+len(set(y))))
print("\nWith %d cells in hidden layer"%q_z)
print("Manual:")
print_result(q_z,l_r=0.0001,iris=False)
print("Scikit:")
scikit_method(X,y,q_z,l_r=0.0001,beta=0.5,k_f=10)

q_z=int((len(X[0])+len(set(y)))*1.5)
print("\nWith %d cells in hidden layer"%q_z)
print("Manual:")
print_result(q_z,l_r=0.0001,iris=False)
print("Scikit:")
scikit_method(X,y,q_z,l_r=0.0001,beta=0.5,k_f=10)





q_z=int((len(X[0])+len(set(y)))/2)

q=0.0001
print("\nWith a %f learning rate"%q)
print("Manual:")
print_result(q_z,l_r=q,iris=False)
print("Scikit:")
scikit_method(X,y,q_z,l_r=q,beta=0.5,k_f=10)


q=0.01
print("\nWith a %f learning rate"%q)
print("Manual:")
print_result(q_z,l_r=q,iris=False)
print("Scikit:")
scikit_method(X,y,q_z,l_r=q,beta=0.5,k_f=10)

q=0.000001
print("\nWith a %f learning rate"%q)
print("Manual:")
print_result(q_z,l_r=q,iris=False)
print("Scikit:")
scikit_method(X,y,q_z,l_r=q,beta=0.5,k_f=10)


