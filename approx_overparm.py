'''
commands run


tensorboard --logdir=/tmp/mdl_logs
tensorboard --logdir=/tmp/mdl_logs
'''

import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import maps

import pdb

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('use_tb', True, 'use tensorboard or not')
tf.app.flags.DEFINE_string('tb_loc_train', './tmp/train', 'tensorboard location')

tf.app.flags.DEFINE_integer('lb', 0, 'lower bound for where to choose data points')
tf.app.flags.DEFINE_integer('ub', 1, 'upper bound for where to choose data points')
tf.app.flags.DEFINE_integer('N', 5, 'N is number of data points')

def poly_kernel_matrix( x,D ):
    N = len(x)
    Kern = np.zeros( (N,D+1) )
    for n in range(N):
        for d in range(D+1):
            Kern[n,d] = x[n]**d;
    return Kern

def get_batch(X,Y,M):
    N = len(Y)
    valid_indices = np.array( range(N) )
    batch_indices = np.random.choice(valid_indices,size=M,replace=False)
    batch_xs = X[batch_indices,:]
    batch_ys = Y[batch_indices]
    return batch_xs, batch_ys

def main(argv=None):
    use_tb = FLAGS.use_tb
    tb_loc_train = FLAGS.tb_loc_train
    ##
    lb, ub = FLAGS.lb, FLAGS.ub
    N = FLAGS.N
    Degree_true= 4
    D_true = Degree_true+1
    D_sgd = Degree_true+1
    D_mdl = Degree_true
    B=1000
    nb_iter = 1
    report_error_freq = nb_iter/4
    ##
    x = np.linspace(FLAGS.lb,FLAGS.ub,5)
    y = np.array([0,1,0,-1,0]).transpose()
    X_true = poly_kernel_matrix( x,Degree_true )
    c_true = np.dot(np.linalg.pinv(X_true),y)
    ##
    graph = tf.Graph()
    with graph.as_default():
        X = tf.placeholder(tf.float32, [None, D_true])
        Y = tf.placeholder(tf.float32, [None])
        w = tf.Variable( tf.zeros([D_sgd,1]) )
        f = X*w
        loss = tf.reduce_sum(Y - f)
        #
        M = 5
        train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)
    #
    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        if use_tb:
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(tb_loc_train,graph)
            fetches_loss = {'loss':loss, 'merged':merged}
        else:
            fetches_loss = {'loss':loss}
        # Train
        for i in range(50000):
            batch_xs, batch_ys = get_batch(X_true,y,M)
            sess.run(train_step, feed_dict={X: batch_xs, Y: batch_ys})
            if i % report_error_freq == 0:
                if use_tb:
                    train_writer.add_summary(fetches_loss['merged'], i)
        #
        c_sgd = w.eval()
        print('c_sgd: ')
        print(c_sgd)
        print('c_true: ')
        print(c_true.shape)
        c_true.shape = (len(c_true),1)
        print(c_true)
        print(' c_sgd - c_true', np.linalg.norm(c_sgd - c_true,2))

if __name__ == '__main__':
    start_time = time.time()
    #tf.app.run()
    main()
    seconds = (time.time() - start_time)
    minutes = seconds/ 60
    hours = minutes/ 60
    print("--- %s seconds ---" % seconds )
    print("--- %s minutes ---" % minutes )
    print("--- %s hours ---" % hours )
    print('\a')
