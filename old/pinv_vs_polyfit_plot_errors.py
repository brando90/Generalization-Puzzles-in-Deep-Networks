import numpy as np
from sklearn.preprocessing import PolynomialFeatures

import scipy
#from scipy import linalg

import pdb

import matplotlib.pyplot as plt

def l2_loss(y,y_):
    N = y.shape[0]
    return (1/N)*np.linalg.norm(y-y_)

def get_errors(lb,ub,degree_mdl,N,f_target):
    ## some parameters
    D0=1
    ## evaluate target_f on x_points
    X = np.linspace(lb,ub,N) # [N,]
    Y = f_target(X) # [N,]
    # get pinv solution
    poly_feat = PolynomialFeatures(degree=degree_mdl)
    Kern = poly_feat.fit_transform( X.reshape(N,D0) ) # low degrees first [1,x,x**2,...]
    #c_pinv = np.dot(np.linalg.pinv( Kern ), Y)
    c_pinv = np.dot(scipy.linalg.pinv( Kern ), Y)
    ## get polyfit solution
    c_polyfit = np.polyfit(X,Y,degree_mdl)[::-1] # need to reverse to get low degrees first [1,x,x**2,...]
    ##
    #c_lstsq,_,_,_ = np.linalg.lstsq(Kern,Y.reshape(N,1))
    ##
    #print('lb,ub = {} '.format((lb,ub)))
    #print('differences with c_pinv')
    #print( '||c_pinv-c_pinv||^2 = {}'.format( np.linalg.norm(c_pinv-c_pinv) ))
    #print( '||c_pinv-c_polyfit||^2 = {}'.format( np.linalg.norm(c_pinv-c_polyfit) ))
    #print( '||c_pinv-c_lstsq||^2 = {}'.format( np.linalg.norm(c_pinv-c_lstsq) ))
    ##
    #print('differences with c_polyfit')
    #print( '||c_polyfit-c_pinv||^2 = {}'.format( np.linalg.norm(c_polyfit-c_pinv) ))
    #print( '||c_polyfit-c_polyfit||^2 = {}'.format( np.linalg.norm(c_polyfit-c_polyfit) ))
    #print( '||c_polyfit-c_lstsq||^2 = {}'.format( np.linalg.norm(c_polyfit-c_lstsq) ))
    ##
    #print('differences with c_lstsq')
    #print( '||c_lstsq-c_pinv||^2 = {}'.format( np.linalg.norm(c_lstsq-c_pinv) ))
    #print( '||c_lstsq-c_polyfit||^2 = {}'.format( np.linalg.norm(c_lstsq-c_polyfit) ))
    #print( '||c_lstsq-c_lstsq||^2 = {}'.format( np.linalg.norm(c_lstsq-c_lstsq) ))
    ##
    #print('Data set errors')
    y_polyfit = np.dot(Kern,c_polyfit)
    #print( 'J_data(c_polyfit) = {}'.format( l2_loss(y_polyfit,Y) ) )
    y_pinv = np.dot(Kern,c_pinv)
    #print( 'J_data(c_pinv) = {}'.format( l2_loss(y_pinv,Y) ) )
    #y_lstsq = np.dot(Kern,c_lstsq)
    y_lstsq = -1*np.dot(Kern,c_pinv)
    #print( 'J_data(c_lstsq) = {}'.format( l2_loss(y_lstsq,Y) ) )
    return l2_loss(y_polyfit,Y),l2_loss(y_pinv,Y),l2_loss(y_lstsq,Y)

####
bounds = np.linspace(1,250,250)
degree_mdl = 100
N=80
D0 = 1
## target function
freq_cos = 0.3
freq_sin = 0.5
#f_target = lambda x: np.exp(-x**2)*np.cos(freq_sin*2*np.pi*x)
#f_target = lambda x: x**2+np.exp(-x**2)*np.cos(freq_sin*2*np.pi*x)
degree_synthetic_poly = 50
def poly(x):
    poly_feat = PolynomialFeatures(degree=degree_synthetic_poly)
    Kern = poly_feat.fit_transform( x.reshape(N,D0) ) # low degrees first [1,x,x**2,...]
    nb_monomials_data = Kern.shape[1]
    #c_mdl = np.arange(1,nb_monomials_data+1).reshape((nb_monomials_data,1))+np.random.normal(loc=3.0,scale=1.0,size=(nb_monomials_data,1))
    c_mdl = np.arange(1,nb_monomials_data+1).reshape((nb_monomials_data,1))
    y = np.dot(Kern,c_mdl)
    print(y.shape)
    return y.reshape((N,))
f_target = lambda x: np.exp(-x**2)*np.sin(freq_sin*2*np.pi*x)
f_target = lambda x: np.cos(freq_cos*2*np.pi*x)+np.sin(freq_sin*2*np.pi*x)
#f_target = lambda x: poly(x)
#f_target = lambda x: poly(x)+np.exp(-x**2)*np.sin(freq_sin*2*np.pi*x)
##
errors_polyfit = []
errors_pinv = []
errors_lstsq = []
for bound in bounds:
    error_polyfit,error_pinv,error_lstsq = get_errors(lb=-bound,ub=bound,degree_mdl=degree_mdl,N=N,f_target=f_target)
    ##
    errors_polyfit.append(error_polyfit)
    errors_pinv.append(error_pinv)
    errors_lstsq.append(error_lstsq)
##
p_polyfit, = plt.plot(bounds,errors_polyfit)
p_pinv, = plt.plot(bounds,errors_pinv)
#p_lstsq, = plt.plot(bounds,errors_lstsq)
#plt.legend([p_polyfit,p_pinv,p_lstsq], ['polyfit','pinv','lstsq'])
plt.legend([p_polyfit,p_pinv], ['polyfit','pinv'])
##
plt.xlabel('bound')
plt.ylabel('least squares error')
plt.title('Least squres errors for models, degree = {}'.format(degree_mdl))


#plot_target_function(c_mdl,X_train,Y_train,lb=lb,ub=ub,f_2_imitate=f_2_imitate)

plt.show()
