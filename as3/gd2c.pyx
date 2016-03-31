import numpy as np

def gradient_descent_two_class(X,y):
	cdef float summ,step, diff
	cdef float[1000] w, w_o
	w = [0.01]*len(X[0])
	diff=10
	step=0.0005
	while(diff>1e-3):
		summ=0
		for i in range(0,len(y)):
			summ+=np.dot(h(X[i],w)-y[i],X[i])
		w_o = w
		w = np.subtract(w,step*summ)
		diff=np.average(np.absolute(w_o-w))
		print (diff)
	return w

def h(x,theta):
	cdef float h
	h = - np.dot(np.transpose(theta),x)
	h = np.exp(h)
	h= 1/(1+h)
	return h