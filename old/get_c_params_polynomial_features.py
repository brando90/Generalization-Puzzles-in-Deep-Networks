from sklearn.preprocessing import PolynomialFeatures

import numpy as np

import pdb

#TODO

## data set
N,D0 = 5, 2
X = np.random.rand(N,D0) # [N,D0]
Y = np.zeros((N,1))
for i in range(N):
    x,y = X[i,0], X[i,1]
    Y[i] = np.sin(2*np.pi*(x+y))

## model
degree_mdl = 10
poly_feat = PolynomialFeatures(degree=degree_mdl)
mdl = poly_feat.fit(X=X,y=y)

pdb.set_trace()
# get pinv solution
#poly_feat = PolynomialFeatures(degree=degree_mdl)
#ern = poly_feat.fit_transform( X.reshape(N,D0) ) # low degrees first [1,x,x**2,...]
#c_pinv = np.dot(np.linalg.pinv( Kern ), Y)
