import numpy as np

from sklearn.preprocessing import normalize
a = np.array([[1,2,3],[4,5,6]])
print(a[0]+a[1])
data = normalize(a, axis=0, norm='l1')
print(data)