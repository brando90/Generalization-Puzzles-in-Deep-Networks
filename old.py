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

def get_RLS_soln( X,Y,lambda_rls):
    N,D = X.shape
    XX_lI = np.dot(X.transpose(),X) + lambda_rls*N*np.identity(D)
    w = np.dot( np.dot( np.linalg.inv(XX_lI), X.transpose() ), Y)
    return w

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
    N = 5
    Degree = 100
    D_sgd = Degree+1
    D_pinv = Degree+1
    D_rls = D_pinv
    B=1000
    nb_iter = 100000
    lambda_rls = 0.1
    #report_error_freq = nb_iter/4
    ##
    np.set_printoptions(suppress=True)
    x = np.linspace(FLAGS.lb,FLAGS.ub,5)
    y = np.array([0,1,0,-1,0])
    y.shape = (N,1)
    X_true = poly_kernel_matrix( x,Degree ) # [N, D] = [N, Degree+1]
    c_pinv = np.dot(np.linalg.pinv(X_true),y) # [N,1]
    c_rls = get_RLS_soln(X_true,y,lambda_rls)
    if D_pinv == N:
        D_true = D_pinv
        c_true = c_pinv
    print(X_true)
    print('\ny: ',y)
    print('\nc_pinv: ', c_pinv)
    # c_pinv = np.linalg.lstsq(X_true,y)
    # print(c_pinv[0])
    #pdb.set_trace()
    ##
    graph = tf.Graph()
    with graph.as_default():
        X = tf.placeholder(tf.float32, [None, D_pinv])
        Y = tf.placeholder(tf.float32, [None,1])
        #w = tf.Variable( tf.zeros([D_sgd,1]) )
        w = tf.Variable( tf.truncated_normal([D_sgd,1],mean=0.0,stddev=1.0) )
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
        M = 2
        eta = 0.02
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate=eta, global_step=global_step,
            decay_steps=nb_iter/2, decay_rate=1, staircase=True)
        train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
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
        c_pinv.shape = (len(c_pinv),1)
        #
        print()
        print('c_sgd: ')
        print(c_sgd)
        print('c_pinv: ')
        print(c_pinv)
        print('\n||c_sgd - c_pinv|| = ', np.linalg.norm(c_sgd - c_pinv,2))
        #
        print('norm(c_sgd): ', np.linalg.norm(c_sgd))
        print('norm(c_pinv): ', np.linalg.norm(c_pinv))
        print('norm(c_rls): ', np.linalg.norm(c_rls))
        #
        print('Degree={}\n nb_iter={}\n batch-size= {} \n learning_rate= {}\n'.format(Degree,nb_iter,M,eta) )
        #
        print(' J(c_sgd) = ', sess.run(fetches=loss, feed_dict={X: X_true, Y: y}) )
        Xc_pinv = np.dot(X_true,c_pinv)
        print( ' J(c_pinv) = ',(1/N)*(np.linalg.norm(y-Xc_pinv)**2) )
        Xc_rls = np.dot(X_true,c_rls)
        print( ' J(c_rls) = ',(1/N)*(np.linalg.norm(y-Xc_rls)**2) )
        #
        print('\a')
        ##
        x_horizontal = np.linspace(lb,ub,1000)
        X_plot = poly_kernel_matrix(x_horizontal,D_sgd-1)
        #plots
        p_sgd, = plt.plot(x_horizontal, np.dot(X_plot,c_sgd))
        p_pinv, = plt.plot(x_horizontal, np.dot(X_plot,c_pinv))
        p_rls, = plt.plot(x_horizontal, np.dot(X_plot,c_rls))
        p_data, = plt.plot(x,y,'ro')
        #
        plt.legend([p_sgd,p_pinv,p_rls,p_data],['sgd curve degree='+str(D_sgd-1),'min norm (pinv) degree='+str(D_pinv-1),'rls regularization lambda={} degree={}'.format(lambda_rls,D_rls-1),'data points'])
        plt.ylabel('f(x)')
        plt.show()

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
