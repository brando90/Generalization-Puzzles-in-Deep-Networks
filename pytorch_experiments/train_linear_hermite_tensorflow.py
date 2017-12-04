import time
start_time = time.time()

import pdb

import numpy as np
from numpy.polynomial.hermite import hermvander

import random

import tensorflow as tf

def get_batch(X,Y,M):
    N = len(Y)
    valid_indices = np.array( range(N) )
    batch_indices = np.random.choice(valid_indices,size=M,replace=False)
    batch_xs = X[batch_indices,:]
    batch_ys = Y[batch_indices]
    return batch_xs, batch_ys

##
D0=1
logging_freq = 100
## SGD params
M = 2
eta = 0.1
#eta = lambda i: eta/(i**0.6)
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
print(f'Kern_train.shape={Kern_train.shape}')
Kern_train = Kern_train.reshape(N_train,Kern_train.shape[1])
##
Kern_train_pinv = np.linalg.pinv( Kern_train )
c_pinv = np.dot(Kern_train_pinv, Y_train)
nb_terms = c_pinv.shape[0]
##
condition_number_hessian = np.linalg.cond(Kern_train)
##
graph = tf.Graph()
with graph.as_default():
    X = tf.placeholder(tf.float32, [None, nb_terms])
    Y = tf.placeholder(tf.float32, [None,1])
    w = tf.Variable( tf.zeros([nb_terms,1]) )
    #w = tf.Variable( tf.truncated_normal([Degree_mdl,1],mean=0.0,stddev=1.0) )
    #w = tf.Variable( 1000*tf.ones([Degree_mdl,1]) )
    ##
    f = tf.matmul(X,w) # [N,1] = [N,D] x [D,1]
    #loss = tf.reduce_sum(tf.square(Y - f))
    loss = tf.reduce_sum( tf.reduce_mean(tf.square(Y-f), 0))
    l2loss_tf = (1/N_train)*2*tf.nn.l2_loss(Y-f)
    ##
    learning_rate = eta
    #global_step = tf.Variable(0, trainable=False)
    #learning_rate = tf.train.exponential_decay(learning_rate=eta, global_step=global_step,decay_steps=nb_iter/2, decay_rate=1, staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
    with tf.Session(graph=graph) as sess:
        Y_train = Y_train.reshape(N_train,1)
        tf.global_variables_initializer().run()
        # Train
        for i in range(nb_iter):
            #if i % (nb_iter/10) == 0:
            if i % (nb_iter/10) == 0 or i == 0:
                current_loss = sess.run(fetches=loss, feed_dict={X: Kern_train, Y: Y_train})
                print(f'tf: i = {i}, current_loss = {current_loss}')
            ## train
            batch_xs, batch_ys = get_batch(Kern_train,Y_train,M)
            sess.run(train_step, feed_dict={X: batch_xs, Y: batch_ys})
print(f'condition_number_hessian = {condition_number_hessian}')
print('\a')
#### PLOTTING
# X_plot = hermvander(x_horizontal,Degree_mdl)
# X_plot = X_plot.reshape(1000,X_plot.shape[2])
# X_plot_pytorch = Variable( torch.FloatTensor(X_plot), requires_grad=False)
# ##
# fig1 = plt.figure()
# ##plots objs
# p_sgd, = plt.plot(x_horizontal, [ float(f_val) for f_val in mdl_sgd.forward(X_plot_pytorch).data.numpy() ])
# p_pinv, = plt.plot(x_horizontal, np.dot(X_plot,c_pinv))
# p_data, = plt.plot(X_train,Y_train,'ro')
# # legend
# nb_terms = c_pinv.shape[0]
# legend_mdl = f'SGD solution standard parametrization, number of monomials={nb_terms}, batch-size={M}, iterations={nb_iter}, step size={eta}'
# plt.legend(
#        [p_sgd,p_pinv,p_data],
#        [legend_mdl,f'linear algebra soln, number of monomials={nb_terms}',f'data points = {N_train}']
#    )
# plt.xlabel('x'), plt.ylabel('f(x)')
# plt.show()
## REPORT TIMES
seconds = (time.time() - start_time)
minutes = seconds/ 60
hours = minutes/ 60
print("--- %s seconds ---" % seconds )
print("--- %s minutes ---" % minutes )
print("--- %s hours ---" % hours )
#plt.show()
