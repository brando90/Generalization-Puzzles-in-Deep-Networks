import numpy as np
from sklearn.preprocessing import PolynomialFeatures

## some parameters
lb,ub = -50,50
N=20000
D0=1
degree_mdl = 80
## target function
freq_cos = 2
f_2_imitate = lambda x: np.cos(freq_cos*2*np.pi*x)
## evaluate target_f on x_points
X = np.linspace(lb,ub,N) # [N,]
Y = f_2_imitate(X) # [N,]
# get pinv solution
poly_feat = PolynomialFeatures(degree=degree_mdl)
Kern = poly_feat.fit_transform( X.reshape(N,D0) ) # low degrees first [1,x,x**2,...]
c_pinv = np.dot(np.linalg.pinv( Kern ), Y)
## get polyfit solution
c_polyfit = np.polyfit(X,Y,degree_mdl)[::-1] # need to reverse to get low degrees first [1,x,x**2,...]
##
print('||c_pinv-c_polyfit||^2 = {}'.format( np.linalg.norm(c_pinv-c_polyfit) ))
