import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import pdb

## some parameters
degree_target = 25
N_train = degree_target+1
lb,ub = -2000,2000
x = np.linspace(lb,ub,N_train)
## generate target polynomial model
freq_cos = 5
y_cos = np.cos(2*np.pi*freq_cos*x)
c_polyfit = np.polyfit(x,y_cos,degree_target)[::-1] ## needs to me reverse to get highest power last
## generate kernel matrix
poly_feat = PolynomialFeatures(degree=degree_target)
K = poly_feat.fit_transform(x.reshape(N_train,1)) # generates degree 0 first
## get target samples of the function
y = np.dot(K,c_polyfit)
## get pinv approximation of c_polyfit
c_pinv = np.dot( np.linalg.pinv(K), y)
## get Gaussian-Elminiation approximation of c_polyfit
c_GE = np.linalg.solve(K,y)
## get inverse matrix approximation of c_polyfit
i = np.linalg.inv(K)
c_mdl_i = np.dot(i,y)
##
c_lstsq,_,_,_ = np.linalg.lstsq(K,y_cos.reshape(N_train,1))
## check rank to see if its truly invertible
print('rank(K) = {}'.format( np.linalg.matrix_rank(K) ))
## comapre parameters
print('--c_polyfit')
print('||c_polyfit-c_GE||^2 = {}'.format( np.linalg.norm(c_polyfit-c_GE) ))
print('||c_polyfit-c_pinv||^2 = {}'.format( np.linalg.norm(c_polyfit-c_pinv) ))
print('||c_polyfit-c_mdl_i||^2 = {}'.format( np.linalg.norm(c_polyfit-c_mdl_i) ))
print('||c_polyfit-c_lstsq||^2 = {}'.format( np.linalg.norm(c_polyfit-c_lstsq) ))
print('||c_polyfit-c_polyfit||^2 = {}'.format( np.linalg.norm(c_polyfit-c_polyfit) ))
##
print('--c_GE')
print('||c_GE-c_GE||^2 = {}'.format( np.linalg.norm(c_GE-c_GE) ))
print('||c_GE-c_pinv||^2 = {}'.format( np.linalg.norm(c_GE-c_pinv) ))
print('||c_GE-c_mdl_i||^2 = {}'.format( np.linalg.norm(c_GE-c_mdl_i) ))
print('||c_GE-c_lstsq||^2 = {}'.format( np.linalg.norm(c_GE-c_lstsq) ))
print('||c_GE-c_polyfit||^2 = {}'.format( np.linalg.norm(c_GE-c_polyfit) ))
##
print('--c_pinv')
print('||c_pinv-c_GE||^2 = {}'.format( np.linalg.norm(c_pinv-c_GE) ))
print('||c_pinv-c_pinv||^2 = {}'.format( np.linalg.norm(c_pinv-c_pinv) ))
print('||c_pinv-c_mdl_i||^2 = {}'.format( np.linalg.norm(c_pinv-c_mdl_i) ))
print('||c_pinv-c_lstsq||^2 = {}'.format( np.linalg.norm(c_pinv-c_lstsq) ))
print('||c_pinv-c_polyfit||^2 = {}'.format( np.linalg.norm(c_pinv-c_polyfit) ))
##
print('--c_mdl_i')
print('||c_mdl_i-c_GE||^2 = {}'.format( np.linalg.norm(c_mdl_i-c_GE) ))
print('||c_mdl_i-c_pinv||^2 = {}'.format( np.linalg.norm(c_mdl_i-c_pinv) ))
print('||c_mdl_i-c_mdl_i||^2 = {}'.format( np.linalg.norm(c_mdl_i-c_mdl_i) ))
print('||c_mdl_i-c_lstsq||^2 = {}'.format( np.linalg.norm(c_mdl_i-c_lstsq) ))
print('||c_mdl_i-c_polyfit||^2 = {}'.format( np.linalg.norm(c_mdl_i-c_polyfit) ))
##
print('--c_lstsq')
print('||c_lstsq-c_GE||^2 = {}'.format( np.linalg.norm(c_lstsq-c_GE) ))
print('||c_lstsq-c_pinv||^2 = {}'.format( np.linalg.norm(c_lstsq-c_pinv) ))
print('||c_lstsq-c_mdl_i||^2 = {}'.format( np.linalg.norm(c_lstsq-c_mdl_i) ))
print('||c_lstsq-c_lstsq||^2 = {}'.format( np.linalg.norm(c_lstsq-c_lstsq) ))
print('||c_lstsq-c_polyfit||^2 = {}'.format( np.linalg.norm(c_lstsq-c_polyfit) ))
