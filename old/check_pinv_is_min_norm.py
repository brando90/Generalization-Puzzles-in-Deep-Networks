import numpy as np
from scipy import linalg as scipy_linalg

import pdb

N,D = 3,10

## NUMPY PINV: A^+
X = np.random.rand(N,D)
X_np_pinv = np.linalg.pinv(X)

## My/Manual PINV: X^T (X X^T)^-1
# note: thats only the correct equations when X has linearly indepedent rows
XXt_inv = np.linalg.inv(np.dot(X,X.T))
X_my_pinv = np.dot(X.T ,XXt_inv )

## SVD (manual) A^+ = V S^+ U^T
U,ss,V = np.linalg.svd(X,full_matrices=True)
# for S
S = np.zeros((D, N))
ss_inv = np.array( [1/s for s in ss] )
S[:N,:N] = np.diag(ss_inv)
S_inv = S
#S_inv = np.linalg.inv(S)
#
#Sp_Ut = np.dot( S_inv, U)
X_svd_pinv = np.dot(V.T, np.dot( S_inv, U.T))

## Scipt pinv
X_scipy_pinv = scipy_linalg.pinv2(X)

##
print('X_np_pinv.shape', X_np_pinv.shape)
print('X_my_pinv.shape', X_my_pinv.shape)

print('Numpy\'s Pseudoinverse')
print(X_np_pinv)
print('My\'s Pseudoinverse')
print(X_my_pinv)
print('SVD (my manual) Pseudoinverse')
print(X_svd_pinv)
print('Scipy Pseudoinverse')
print(X_scipy_pinv)
