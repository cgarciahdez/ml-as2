import numpy as np

#
#This function loads a file with the name given into to np arrays x and y.
#Parameters: Name of the file to load. Filt : if not -1, filters out only the
#classes in the given set
#Returns: nparray x with feature information and nparray y with label information
def load_file(name, filt=-1):
    print(filt)
    file = open(name,'r')
    x=[]
    y=[]
    for line in file:
        if not line.startswith('#'):
            l=line.split(',')
            xi=[]
            for i in range(1,len(l)):
                xi.append(float(l[i]))
            yi=float(l[-1])
            if not yi == 0 and (filt==-1 or yi in filt):
                y.append(yi)
                x.append(xi)
    return np.array(x), np.array(y)

#
#This function puts together the results of loading all the data
#that is split into different participants data
#Returns: np araay x with feature information and np array y with label information
def put_files_together(filt=-1):
    x,y=load_file("./data/%d.csv"%1,filt)
    for i in range(2,16):
        xi,yi=load_file("./data/%d.csv"%i,filt)
        x=np.concatenate((x,xi))
        y=np.concatenate((y,yi))
    return x,y


def estimate_parameters_uniGDA():
    pass

print (put_files_together((1,2)))

