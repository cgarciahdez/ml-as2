import numpy as np
import cvxopt
from cvxopt import solvers, matrix
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.cross_validation import KFold
from sklearn.datasets import fetch_mldata


#
#This function loads MNIST into to np arrays x and y.
#Params: two class true if only two classes d degree (for mapping)
#Returns: nparray x with feature information and nparray y with label information
def load_data(two_class=True,d=1):
	mnist = fetch_mldata('MNIST original')
	mnist.data.shape
	mnist.target.shape
	np.unique(mnist.target)

	X, y = mnist.data / 255., mnist.target
	if two_class:
		ind = [ k for k in range(len(y)) if (y[k]==0 or y[k]==1) ]
		X=X[ind,:]
		y=y[ind]
	for i in range(len(y)):
		if y[i]==1:
			y[i]=1
		else:
			y[i]=-1


	X,y= np.array(X),np.array(y)
	X, y = shuffle(X, y, random_state=0)
	X,y = X[:500],y[:500]

	y = y.astype(np.double)

	print(X)
	print(y)

	return X,y

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

#This function generates the data to do testing. It can be separable, have uneven distribution of classes,
# and has the option to cluster the data to make it even again.
#Number of examples is also customizable. The separable data is alwas separated by function x1=x2
def generate(separable=True,m=200,more=False,cluster=False):
	x=np.random.uniform(0.0, 1.0, (m,2))
	y=[]
	if separable:
		n=0
		if more:
			n=0.5
		for x_i in x:
			if x_i[0]<x_i[1]+n:
				y.append(-1)
			else:
				y.append(1)
	else:
		n=int(m/2)
		if more:
			n = int(m/6)
		y=[-1]*int(n)
		y=y+[1]*int(m-n)

	if cluster:
		km = KMeans()
		km.n_clusters = n

	x,y = np.array(x),np.array(y)
	x, y = shuffle(x, y, random_state=0)

	return x,y

#
#This function fits the data to the SVM. This SVM accepts the kernel as a function, so it can
#change as long as the function exists and returns k(x_i,x_j). Returns necessary parameters.
#It also allows for user slected c parameter.
#For kenerls, returns support vector information in order to do classifying.		
def SVM(x,y,c=None, kernel=None):
	a, prim = solve_dual(x,y,c,kernel)

	SV = []
	w = 0
	for i in range(len(x)):
		if a[i]>1e-10:
			SV.append(i)
		w += a[i]*y[i]*x[i]

	w_o=0

	for i in SV:
		w_o += y[i]-np.dot(w.T,x[i])

	w_o/=len(SV)

	if kernel is None:
		return w, w_o, SV
	else:
		return w, w_o, SV, x[SV], y[SV], a[SV]

#
#This function decides wether the classyfying is kernel or not
#Depending on the number of params
def classify(x_,p,kernel=None):
	if len(p)>3:
		return classify_kernel(p[3],p[4],p[5],p[1],x_,kernel)
	else:
		return classify_linear(x_,p[0],p[1])

#
#This funcion classifies linear SVM results
def classify_linear(x_,w,w_o):
	if np.dot(w.T,x_)+w_o>0:
		return 1
	return -1

#
#This funcion classifies kernel SVM results
#Takes the kernel as a function and all necessary params
def classify_kernel(x,y,a,w_o,x_,kernel):
	s=0
	for i in range(len(x)):
		s += a[i]*y[i]*kernel(x[i],x_)

	if s+w_o>0:
		return 1
	return -1

#
#This function solves the dual objective to find the prioris and primal objective
#using the CVXOPT library. Solves for both linear or kernel SVM, soft or hard margins.
def solve_dual(x,y,c=None, kernel=None):
	n = len(x[0])
	m = len(x)
	if kernel is None:
		K = np.zeros((m,m))
		for i in range(m):
		    for j in range(m):
		          K[i,j] = np.dot(x[i], x[j])
		P = matrix(np.outer(y, y) * K)
	else:
		K = np.zeros((m,m))
		for i in range(m):
		    for j in range(m):
		          K[i,j] = kernel(x[i], x[j])
		P = matrix(np.outer(y, y) * K)
	A = matrix(y, (1,m), 'd')
	q = np.empty((m))
	q.fill(-1)
	b = matrix(0.0)

	if c is None:
		h = np.zeros((m))
		G = np.zeros((m,m))
		np.fill_diagonal(G,-1)
	else:
		h_2 = np.empty((m))
		h_2.fill(c)
		h_1 = np.zeros((m))
		h = np.concatenate((h_1,h_2))
		G_1 = np.zeros((m,m))
		np.fill_diagonal(G_1,-1)
		G_2 = np.zeros((m,m))
		np.fill_diagonal(G_2,1)
		G = np.concatenate((G_1,G_2),axis=0)

	sol = solvers.qp(matrix(P),matrix(q),matrix(G),matrix(h),A,b)

	alpha = np.ravel(sol['x'])
	prim = sol['primal objective']

	return alpha, prim

#
#This is the gaussian kernel function.
def gaussian_kernel(x_i,x_j,sigma=0.5):
	sigma = global_sigma
	ret = np.exp(-np.linalg.norm(x_i-x_j)**2/2*sigma**2)
	return ret

#
#This is the polynomial kernel function.
def polynomial_kernel(x_i,x_j,q=3):
	q = global_q
	ret = np.dot(x_i.T,x_j)+1
	ret = ret**q
	return ret

#This function plots the data and the support vectors
def plot(X,Y,SV):
	fig  = plt.figure()
	ax = fig.add_subplot(111)
	plt.plot(X[SV,0],X[SV,1],'ys',alpha=0.4)
	for x, y in zip(X,Y):
		if y == 1:
		      ax.scatter(x[0], x[1], alpha=0.8, color="red", marker="o")
		else:
		      ax.scatter(x[0], x[1], alpha=0.8, color="blue", marker="o")
	#for i in SV:
	    #ax.scatter(X[i,0], X[i,1], s=100, alpha = 0.6, color="green")
	
	#ax.plot(range(len(X)))
	plt.axis("tight")
	plt.show()

#
#Computes classifier performance evaluators given the parameters and the featue (x) and
#original label data (y).
#Parameters: x: feature matrix or vector. y: label vector. theta: theta vector.
#Return: confusion matrix, precision, recall, f_measuer and accuracy.
def confusion_matrix_2(X,y,p,kernel=None):
	tp=0
	tn=0
	fp=0
	fn=0
	i=0
	for i in range(0,len(X)):
	    h_=classify(X[i],p,kernel)
	    if h_ == y[i]:
	        if y[i]==1:
	            tn+=1
	        else:
	            tp+=1
	    else:
	        if y[i]==1:
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
#This function performs crossvalidation to find the average training and testing evaluators
#given data , given a degree and a k-factor.
#Parameters: x - vector or matrix, y - vector
#Returns: accuracies
def cross_validation(X,y,c=None,kernel=None,k_f=10,pl=True):
	k=set(y)
	
	traine={'p':0,'r':0,'f_m':0,'a':0}
	teste={'p':0,'r':0,'f_m':0,'a':0}
	if pl:
		p = SVM(X,y,c,kernel)
		plot(X,y,p[2])

	kf = KFold(len(X),k_f)
	for train, test in kf:
		p = SVM(X[train],y[train],c,kernel)
		rest=confusion_matrix_2(X[train],y[train],p,kernel)
		rese=confusion_matrix_2(X[test],y[test],p,kernel)

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

	#
#This function performs crossvalidation to find the average training and testing evaluators
#given data , given a degree and a k-factor.
#Parameters: x - vector or matrix, y - vector
#Returns: accuracies
def cross_validation_sikckit(X,y,c=1.0,kernel='linear',k_f=10):
	
	traine=0
	teste=0
	clf = SVC()
	clf.kernel='linear'
	clf.C=5000


	kf = KFold(len(X),k_f)
	for train, test in kf:
		clf.fit(X[test], y[test])
		traine+=clf.score(X[train],y[train])
		teste+=clf.score(X[test],y[test])

	print("Hard Margins:")
	print ("Average training error for sickit implementation: %f"%(traine/k_f))
	print ("Average testing error for sickit implementation: %f"%(teste/k_f))

	traine=0
	teste=0
	clf = SVC()
	clf.kernel='linear'
	clf.C=1.0

	kf = KFold(len(X),k_f)
	for train, test in kf:
		clf.fit(X[test], y[test])
		traine+=clf.score(X[train],y[train])
		teste+=clf.score(X[test],y[test])

	print("soft margins:")
	print ("Average training error for sickit implementation: %f"%(traine/k_f))
	print ("Average testing error for sickit implementation: %f"%(teste/k_f))

	traine=0
	teste=0
	clf = SVC()
	clf.kernel='poly'
	clf.C=c

	kf = KFold(len(X),k_f)
	for train, test in kf:
		clf.fit(X[test], y[test])
		traine+=clf.score(X[train],y[train])
		teste+=clf.score(X[test],y[test])

	print("polynomial kernel degree 3:")
	print ("Average training error for sickit implementation: %f"%(traine/k_f))
	print ("Average testing error for sickit implementation: %f"%(teste/k_f))

		

	return traine/k_f, teste/k_f

#
#This function prints results in a legible way.
def print_result(separable=True,more=False,m=200,c=None,kernel=None,name=None,pl=True,cluster=False):
	X,y=generate(separable=separable,more=more,m=m,cluster=cluster)
	if name is not None:
		X,y = load_data()


	p=cross_validation(X,y,pl=pl)
	print ("The training performance evaluators are:\n"\
	"Precision: %f\n Recall: %f\n F_measure: %f\n Accuracy: %f\n\n"\
	"The tesing performance evaluators are:\n"\
	"Precision: %f\n Recall: %f\n F_measure: %f\n Accuracy: %f\n\n" % (p[0]['p'],p[0]['r'],p[0]['f_m'],p[0]['a'],p[1]['p'],p[1]['r'],p[1]['f_m'],p[1]['a']))


print("\n\nSeparable hard margin")
print_result(separable=True,more=False,c=None,kernel=None)
print("\n\nInseparable hard margin")
print_result(separable=False,more=False,c=None,kernel=None)

print("\n\nSeparable soft margin")
print_result(separable=True,more=False,c=5,kernel=None)
print("\n\nInseparable soft margin")
print_result(separable=False,more=False,c=5,kernel=None)

print("\n\nSeparable polynomial kernel with 2 degree")
global_q = 2
print_result(separable=True,more=False,c=None,kernel=polynomial_kernel)
print("\n\nSeparable polynomial kernel with 3 degree")
global_q = 3
print_result(separable=True,more=False,c=None,kernel=polynomial_kernel)
print("\n\nSeparable polynomial kernel with 4 degree")
global_q = 4
print_result(separable=True,more=False,c=None,kernel=polynomial_kernel)
print("\n\nInseparable polynomial kernel")
global_q = 3
print_result(separable=False,more=False,c=None,kernel=polynomial_kernel)

print("\n\Polynomial kernel with 3 degree on external dataset 1")
global_q = 3
print_result(separable=True,more=False,c=None,kernel=polynomial_kernel,name="./data/svar-set1.dat.txt",pl=False)



print("\n\nSeparable gaussian kernel with 0.1 sigma")
global_sigma = 0.1
print_result(separable=True,more=False,c=None,kernel=gaussian_kernel)
print("\n\nSeparable gaussian kernel with 0.5 sigma")
global_sigma = 0.5
print_result(separable=True,more=False,c=None,kernel=gaussian_kernel)
print("\n\nSeparable gaussian kernel with 1.0 sigma")
global_sigma = 1.0
print_result(separable=True,more=False,c=None,kernel=gaussian_kernel)
print("\n\nInseparable gaussian kernel")
global_sigma = 0.5
print_result(separable=False,more=False,c=None,kernel=gaussian_kernel)

print("\n\Gaussian kernel with 0.5 sigma on external dataset 2")
global_sigma = 0.5
print_result(separable=True,more=False,c=None,kernel=gaussian_kernel,name="./data/svar-set2.dat.txt",pl=False)


print("\n\nSeparable hard margin with uneven distribution of classes")
print_result(separable=True,more=True,c=None,kernel=None)

print("\n\nSeparable hard margin with uneven distribution of classes after clustering")
print_result(separable=True,more=False,c=None,kernel=None,pl=False,cluster=True)

print("\n\nSickit for separable data:")
x,y = generate(separable=True)
cross_validation_sikckit(x,y)

print("\n\nSickit for inseparable data:")
x,y = generate(separable=False)
cross_validation_sikckit(x,y)



