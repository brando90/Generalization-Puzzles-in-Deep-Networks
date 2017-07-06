'''
commands run


tensorboard --logdir=/tmp/mdl_logs
tensorboard --logdir=/Users/brandomiranda/home_simulation_research/overparametrized_experiments/tmp/
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
tf.app.flags.DEFINE_string('tb_loc', './tmp/', 'tensorboard location')

tf.app.flags.DEFINE_integer('lb', 0, 'lower bound for where to choose data points')
tf.app.flags.DEFINE_integer('ub', 1, 'upper bound for where to choose data points')
tf.app.flags.DEFINE_integer('N', 5, 'N is number of data points')

def delete_old_runs(tb_loc):
    if tf.gfile.Exists(tb_loc):
      tf.gfile.DeleteRecursively(tb_loc)
    tf.gfile.MakeDirs(tb_loc)

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
    tb_loc = FLAGS.tb_loc
    ##
    lb, ub = FLAGS.lb, FLAGS.ub
    N = FLAGS.N
    Degree_true= 4
    D_true = Degree_true+1
    D_sgd = Degree_true+1
    D_mdl = Degree_true
    B=1000
    nb_iter = 40000
    #report_error_freq = nb_iter/4
    ##
    np.set_printoptions(suppress=True)
    x = np.linspace(FLAGS.lb,FLAGS.ub,5)
    y = np.array([0,1,0,-1,0])
    y.shape = (N,1)
    X_true = poly_kernel_matrix( x,Degree_true ) # [N, D] = [N, Degree+1]
    c_true = np.dot(np.linalg.pinv(X_true),y) # [N,1]
    print(X_true)
    print('\ny: ',y)
    print('\nc_true: ', c_true)
    # c_true = np.linalg.lstsq(X_true,y)
    # print(c_true[0])
    #pdb.set_trace()
    ##
    graph = tf.Graph()
    with graph.as_default():
        X = tf.placeholder(tf.float32, [None, D_true])
        Y = tf.placeholder(tf.float32, [None,1])
        w = tf.Variable( tf.zeros([D_sgd,1]) )
        #w = tf.Variable( tf.truncated_normal([D_sgd,1],mean=0.0,stddev=30.0) )
        #
        w_2 = tf.norm(w)
        tf.summary.scalar('norm(w)',w_2)
        #
        f = tf.matmul(X,w) # [N,1] = [N,D] x [D,1]
        #loss = tf.reduce_sum(tf.square(Y - f))
        loss = tf.reduce_sum( tf.reduce_mean(tf.square(Y-f), 0))
        l2loss_tf = (1/N)*2*tf.nn.l2_loss(Y-f)
        tf.summary.scalar('loss', loss)
        #
        var_grad = tf.gradients(loss, [w])
        g_2 = tf.norm(var_grad)
        tf.summary.scalar('norm(g)', g_2)
        #
        M = 5
        train_step = tf.train.GradientDescentOptimizer(0.9).minimize(loss)
    #
    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        #
        l2_np = (1/N)*np.linalg.norm(y -  (np.dot(X_true,w.eval())) )**2
        print('>>l2_np: ',l2_np)
        print('>>l2_loss_val: ', sess.run(l2loss_tf,{X:X_true,Y:y}) )
        print('>>l2loss_tf: ',sess.run(l2loss_tf,{X:X_true,Y:y}))
        #
        if use_tb:
            delete_old_runs(tb_loc)
            #
            merged_summary = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(tb_loc_train,graph)
            train_writer.add_graph(sess.graph)
        # Train
        for i in range(nb_iter):
            batch_xs, batch_ys = get_batch(X_true,y,M)
            batch_xs, batch_ys = X_true, y
            if i % 100 == 0:
                current_loss = sess.run(fetches=loss, feed_dict={X: X_true, Y: y})
                #print('loss: ', current_loss)
                if use_tb:
                    summary = sess.run(fetches=merged_summary, feed_dict={X: X_true, Y: y})
                    train_writer.add_summary(summary,i)
            sess.run(train_step, feed_dict={X: batch_xs, Y: batch_ys})
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
