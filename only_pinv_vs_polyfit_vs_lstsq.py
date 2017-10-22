import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def l2_loss(y,y_):
    N = y.shape[0]
    return (1/N)*np.linalg.norm(y-y_)

## some parameters
lb,ub = -200,200
N=100
D0=1
degree_mdl = 120
## target function
freq_cos = 2
f_target = lambda x: np.cos(freq_cos*2*np.pi*x)
## evaluate target_f on x_points
X = np.linspace(lb,ub,N) # [N,]
Y = f_target(X) # [N,]
# get pinv solution
poly_feat = PolynomialFeatures(degree=degree_mdl)
Kern = poly_feat.fit_transform( X.reshape(N,D0) ) # low degrees first [1,x,x**2,...]
c_pinv = np.dot(np.linalg.pinv( Kern ), Y)
## get polyfit solution
c_polyfit = np.polyfit(X,Y,degree_mdl)[::-1] # need to reverse to get low degrees first [1,x,x**2,...]
##
c_lstsq,_,_,_ = np.linalg.lstsq(Kern,Y.reshape(N,1))
##
print('lb,ub = {} '.format((lb,ub)))
print('differences with c_pinv')
print( '||c_pinv-c_pinv||^2 = {}'.format( np.linalg.norm(c_pinv-c_pinv) ))
print( '||c_pinv-c_polyfit||^2 = {}'.format( np.linalg.norm(c_pinv-c_polyfit) ))
print( '||c_pinv-c_lstsq||^2 = {}'.format( np.linalg.norm(c_pinv-c_lstsq) ))
##
print('differences with c_polyfit')
print( '||c_polyfit-c_pinv||^2 = {}'.format( np.linalg.norm(c_polyfit-c_pinv) ))
print( '||c_polyfit-c_polyfit||^2 = {}'.format( np.linalg.norm(c_polyfit-c_polyfit) ))
print( '||c_polyfit-c_lstsq||^2 = {}'.format( np.linalg.norm(c_polyfit-c_lstsq) ))
##
print('differences with c_lstsq')
print( '||c_lstsq-c_pinv||^2 = {}'.format( np.linalg.norm(c_lstsq-c_pinv) ))
print( '||c_lstsq-c_polyfit||^2 = {}'.format( np.linalg.norm(c_lstsq-c_polyfit) ))
print( '||c_lstsq-c_lstsq||^2 = {}'.format( np.linalg.norm(c_lstsq-c_lstsq) ))
##
print('Data set errors')
y_polyfit = np.dot(Kern,c_polyfit)
print( 'J_data(c_polyfit) = {}'.format( l2_loss(y_polyfit,Y) ) )
y_pinv = np.dot(Kern,c_pinv)
print( 'J_data(c_pinv) = {}'.format( l2_loss(y_pinv,Y) ) )
y_lstsq = np.dot(Kern,c_lstsq)
print( 'J_data(c_lstsq) = {}'.format( l2_loss(y_lstsq,Y) ) )
