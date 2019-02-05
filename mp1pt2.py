import numpy as np
import bisect

a = np.zeros((9,6))
print a
print ""
a[0:2,1:5] = 1
a[7:,1:5] = 1
a[2:7,2:4] = 1
print a
print ""
z = np.zeros((1,6))
b = np.concatenate((z,a), axis=0)
b = np.concatenate((b,z), axis=0)
print b
print ""
c = np.arange(1,67).reshape((11,6))
print c
print ""
d = np.multiply(b,c)
print d
print ""
e = np.unique(d)[1:]
max,min = e.max(), e.min()
print e
print ""
f = [(e[i]-min)/(max-min) for i in range(len(e))]
print f
print ""

nearest = f[bisect.bisect_left(f, 0.25)]
print nearest
