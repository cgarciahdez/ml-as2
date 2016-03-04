import numpy as np

#
#This function loads a file with the name given into to np arrays x and y.
#Parameters: Name of the file to load
#Returns: nparray x with feature information and nparray y with label information
def load_file(name):
    file = open(name,'r')
    x=[]
    y=[]
    for line in file:
        if not line.startswith('#'):
            l=line.split(',')
            xi=[]
            for i in range(1,len(l)):
                xi.append(float(l[i]))
            x.append(xi)
            yi=float(l[-1])
            if not yi == 0:
            	y.append(yi)
    return np.array(x), np.array(y)

def put_files_together():
	x,y=load_file("./data/%d.csv"%1)
	for i in range(2,16):
		xi,yi=load_file("./data/%d.csv"%i)
		x=np.concatenate((x,xi))
		y=np.concatenate((y,yi))
	return x,y

print (put_files_together())

