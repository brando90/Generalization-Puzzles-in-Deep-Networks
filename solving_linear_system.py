import numpy as np
from sklearn.preprocessing import PolynomialFeatures

## some parameters
degree_target = 25
N_train = degree_target+1
lb,ub = -2000,2000
x = np.linspace(lb,ub,N_train)
## generate target polynomial model
freq_cos = 5
y_cos = np.cos(2*np.pi*freq_cos*x)
c_target = np.polyfit(x,y_cos,degree_target)[::-1] ## needs to me reverse to get highest power last
## generate kernel matrix
poly_feat = PolynomialFeatures(degree=degree_target)
K = poly_feat.fit_transform(x.reshape(N_train,1)) # generates degree 0 first
## get target samples of the function
y = np.dot(K,c_target)
## get pinv approximation of c_target
c_pinv_target = np.dot( np.linalg.pinv(K), y)
## get Gaussian-Elminiation approximation of c_target
c_GE = np.linalg.solve(K,y)
## get inverse matrix approximation of c_target
i = np.linalg.inv(K)
c_mdl_i = np.dot(i,y)
## check rank to see if its truly invertible
print('rank(K) = {}'.format( np.linalg.matrix_rank(K) ))
## comapre parameters
print('||c_GE-c_target||^2 = {}'.format( np.linalg.norm(c_pinv_target-c_target) ))
print('||c_GE-c_target||^2 = {}'.format( np.linalg.norm(c_GE-c_target) ))
print('||c_GE-c_target||^2 = {}'.format( np.linalg.norm(c_mdl_i-c_target) ))
