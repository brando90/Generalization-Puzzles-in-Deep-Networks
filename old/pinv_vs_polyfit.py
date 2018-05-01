import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import scipy.integrate as integrate

import matplotlib.pyplot as plt

def plot_target_function(c_mdl,X,Y,lb,ub,f_target,legend):
    fig1 = plt.figure()
    deg = c_mdl.shape[0]-1
    ## plotting data (note this is NOT training data)
    N=5000
    x_plot_points = np.linspace(lb,ub,N).reshape(N,1) # [N,1]
    ## evaluate the model given on plot points
    poly_feat = PolynomialFeatures(degree=deg)
    Kern_plot_points = poly_feat.fit_transform(x_plot_points)
    y_plot_points = np.dot(Kern_plot_points,c_mdl)
    #
    x_for_f = np.linspace(lb,ub,30000)
    #pdb.set_trace()
    y_for_f = f_target( x_for_f )
    #
    p_mdl, = plt.plot(x_plot_points,y_plot_points)
    p_f_target, = plt.plot(x_for_f,y_for_f)
    #p_training_data, = plt.plot(X,Y,'ro')
    #plt.legend([p_mdl,p_f_target,p_training_data], ['Target function f(x) of degree {}'.format(deg),'f trying to imitate','data points'])
    plt.legend([p_mdl,p_f_target], ['approximation functio {}'.format(legend),'Target function f(x)'])
    ##
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Model has degree {}'.format(deg))

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
def f(c_mdl,x):
    degree_mdl = c_mdl.shape[0]-1
    poly_feat = PolynomialFeatures(degree=degree_mdl)
    #Kern = poly_feat.fit_transform( x.reshape(x.shape[0],x.shape[1]) )
    Kern = poly_feat.fit_transform( x )
    return np.dot(Kern,c_mdl)

def L2_func_diff(f1,f2,lb,ub):
    f_diff_squared = lambda x: (f1(x)-f2(x))**2
    func_diff = integrate.quad(f_diff_squared, lb,ub)
    return func_diff

f_pinv = lambda x: f(c_pinv,x)
f_polyfit = lambda x: f(c_polyfit,x)
f_lstsq = lambda x: f(c_lstsq,x)
diff_f_target_f_polyfit = L2_func_diff(f_polyfit,f_target,lb,ub)
diff_f_target_f_pinv = L2_func_diff(f_pinv,f_target,lb,ub)
diff_f_target_f_lstsq = L2_func_diff(f_lstsq,f_target,lb,ub)
##
print('|| f_polyfit - f_target ||^2 = {}'.format(diff_f_target_f_polyfit))
print('|| f_pinv - f_target ||^2 = {}'.format(diff_f_target_f_pinv))
print('|| f_lstsq - f_target ||^2 = {}'.format(diff_f_target_f_lstsq))
##
# print('differences with c_polyfit')
# print( '||X*c_pinv - X*c_polyfit||^2 = {}'.format( np.linalg.norm(np.dot(Kern,c_pinv)-np.dot(Kern,c_polyfit)) ))
# print( '||X*c_pinv - X*c_polyfit||^2 = {}'.format( np.linalg.norm(np.dot(Kern,c_pinv)-np.dot(Kern,c_polyfit)) ))
# print( '||X*c_pinv - X*c_polyfit||^2 = {}'.format( np.linalg.norm(np.dot(Kern,c_pinv)-np.dot(Kern,c_polyfit)) ))
##
print('Data set errors')
y_polyfit = np.dot(Kern,c_polyfit)
print( 'J_data(c_polyfit) = {}'.format( l2_loss(y_polyfit,Y) ) )
y_pinv = np.dot(Kern,c_pinv)
print( 'J_data(c_pinv) = {}'.format( l2_loss(y_pinv,Y) ) )
y_lstsq = np.dot(Kern,c_lstsq)
print( 'J_data(c_lstsq) = {}'.format( l2_loss(y_lstsq,Y) ) )

plot_target_function(c_polyfit,X,Y,lb,ub,f_target,legend='f_polyfit')
plot_target_function(c_pinv,X,Y,lb,ub,f_target,legend='f_pinv')
plot_target_function(c_lstsq,X,Y,lb,ub,f_target,legend='f_lstsq')
plt.show()
