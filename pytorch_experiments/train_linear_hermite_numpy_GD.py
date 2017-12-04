import time
start_time = time.time()

import numpy as np
from numpy.polynomial.hermite import hermvander

import random


##
D0=1
logging_freq = 100
#dtype = torch.cuda.FloatTensor
dtype = torch.FloatTensor
## SGD params
M = 2
eta = lambda i: 0.1/(i**0.6)
nb_iter = 500*10
##
lb,ub = 0,1
freq_sin = 4 # 2.3
f_target = lambda x: np.sin(2*np.pi*freq_sin*x)
N_train = 10
X_train = np.linspace(lb,ub,N_train)
Y_train = f_target(X_train).reshape(N_train,1)
x_horizontal = np.linspace(lb,ub,1000).reshape(1000,1)
## degree of mdl
Degree_mdl = N_train-1
## Hermite
Kern_train = hermvander(X_train,Degree_mdl)
##
Kern_train_pinv = np.linalg.pinv( Kern_train )
c_pinv = np.dot(Kern_train_pinv, Y_train)
##
condition_number_hessian = np.linalg.cond(Kern_train)
## Make polynomial Kernel
for t in range(1:nb_iter+1):
    
## PRINT ERRORS
from plotting_utils import *

train_error_pinv = (1/N_train)*(np.linalg.norm(Y_train-np.dot(Kern_train,c_pinv))**2)
print('\n-----------------')
print(f'N_train={N_train}')
print(f'train_error_pinv = {train_error_pinv}')
print(f'final_sgd_error = {final_sgd_error}')

print(f'condition_number_hessian = {condition_number_hessian}')
print('\a')
#### PLOTTING
X_plot = hermvander(x_horizontal,Degree_mdl)
X_plot = X_plot.reshape(1000,X_plot.shape[2])
X_plot_pytorch = Variable( torch.FloatTensor(X_plot), requires_grad=False)
##
fig1 = plt.figure()
##plots objs
p_sgd, = plt.plot(x_horizontal, [ float(f_val) for f_val in mdl_sgd.forward(X_plot_pytorch).data.numpy() ])
p_pinv, = plt.plot(x_horizontal, np.dot(X_plot,c_pinv))
p_data, = plt.plot(X_train,Y_train,'ro')
# legend
nb_terms = c_pinv.shape[0]
legend_mdl = f'SGD solution standard parametrization, number of monomials={nb_terms}, batch-size={M}, iterations={nb_iter}, step size={eta}'
plt.legend(
       [p_sgd,p_pinv,p_data],
       [legend_mdl,f'linear algebra soln, number of monomials={nb_terms}',f'data points = {N_train}']
   )
#
plt.xlabel('x'), plt.ylabel('f(x)')
plt.show()
## REPORT TIMES
seconds = (time.time() - start_time)
minutes = seconds/ 60
hours = minutes/ 60
print("--- %s seconds ---" % seconds )
print("--- %s minutes ---" % minutes )
print("--- %s hours ---" % hours )
#plt.show()
