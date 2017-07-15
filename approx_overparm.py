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

#tf.app.flags.DEFINE_integer('lb', 0, 'lower bound for where to choose data points')
#tf.app.flags.DEFINE_integer('ub', 1, 'upper bound for where to choose data points')
#tf.app.flags.DEFINE_integer('N', 5, 'N is number of data points')

def get_c_true(Degree_true,lb=0,ub=1):
    x = np.linspace(lb,ub,5)
    y = np.array([0,1,0,-1,0])
    y.shape = (5,1)
    X_mdl = poly_kernel_matrix( x,Degree_true ) # [N, D] = [N, Degree_mdl+1]
    c_true = np.dot(np.linalg.pinv(X_mdl),y) # [N,1]
    return c_true

def get_data_set_points(c_true,x,Degree_truth=4):
    N = len(x)
    X = poly_kernel_matrix(x,Degree_truth)
    Y = np.dot(X,c_true)
    Y.shape = (N,1)
    return X,Y

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

def get_old():
    x = np.linspace(0,1,5)
    y = np.array([0,1,0,-1,0])
    y.shape = (N,1)
    X_true = poly_kernel_matrix( x,Degree ) # [N, D] = [N, Degree+1]
    return X_true, y

def main(argv=None):
    start_time = time.time()
    #
    np.set_printoptions(suppress=True)
    use_tb = FLAGS.use_tb
    tb_loc_train = FLAGS.tb_loc_train
    tb_loc = FLAGS.tb_loc
    lb, ub = 0, 1
    ## true facts of the data set
    #B=10000
    N = 5
    Degree_true = 4
    D_true = Degree_true+1
    ## mdl degree and D
    Degree_mdl = 100
    D_sgd = Degree_mdl+1
    D_pinv = Degree_mdl+1
    D_rls = D_pinv
    ## sgd
    M = 2
    eta = 0.002
    nb_iter = 6*180000
    lambda_rls = 0.0001
    ##
    x_true = np.linspace(lb,ub,N) # the real data points
    #y = np.array([0,1,0,-1,0])
    c_true = get_c_true(Degree_true,lb,ub)
    X_true,y = get_data_set_points(c_true,x_true) # maps to the real feature space
    #pdb.set_trace()
    #y = np.sin(2*np.pi*x_true)
    y.shape = (N,1)
    ## get linear algebra mdls
    X_mdl = poly_kernel_matrix(x_true,Degree_mdl) # maps to the feature space of the model
    c_pinv = np.dot(np.linalg.pinv(X_mdl),y) # [D_pinv,1]
    c_rls = get_RLS_soln(X_mdl,y,lambda_rls) # [D_pinv,1]
    ##
    graph = tf.Graph()
    with graph.as_default():
        X = tf.placeholder(tf.float32, [None, D_sgd])
        Y = tf.placeholder(tf.float32, [None,1])
        w = tf.Variable( tf.zeros([D_sgd,1]) )
        #w = tf.Variable( tf.truncated_normal([D_sgd,1],mean=0.0,stddev=1.0) )
        #w = tf.Variable( 1000tf.ones([D_sgd,1]) )
        #
        w_2 = tf.norm(w)
        tf.summary.scalar('norm(w)',w_2)
        #
        f = tf.matmul(X,w) # [N,1] = [N,D] x [D,1]
        #loss = tf.reduce_sum(tf.square(Y - f))
        loss = tf.reduce_sum( tf.reduce_mean(tf.square(Y-f), 0))
        l2loss_tf = (1/N)*2*tf.nn.l2_loss(Y-f)
        #loss = l2loss_tf
        tf.summary.scalar('loss', loss)
        #
        var_grad = tf.gradients(loss, [w])
        g_2 = tf.norm(var_grad)
        tf.summary.scalar('norm(g)', g_2)
        #
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate=eta, global_step=global_step,
            decay_steps=nb_iter/2, decay_rate=1, staircase=True)
        train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
    #
    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        #
        l2_np = (1/N)*np.linalg.norm(y -  (np.dot(X_mdl,w.eval())) )**2
        print()
        print('>>norm(y): ', (1/N)*np.linalg.norm(y)**2)
        print('>>l2_np: ',l2_np)
        print('>>l2_loss_val: ', sess.run(l2loss_tf,{X:X_mdl,Y:y}) )
        print('>>l2loss_tf: ',sess.run(l2loss_tf,{X:X_mdl,Y:y}))
        #
        if use_tb:
            delete_old_runs(tb_loc)
            #
            merged_summary = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(tb_loc_train,graph)
            train_writer.add_graph(sess.graph)
        # Train
        for i in range(nb_iter):
            batch_xs, batch_ys = get_batch(X_mdl,y,M)
            #batch_xs, batch_ys = X_mdl, y
            if i % 500 == 0:
                current_loss = sess.run(fetches=loss, feed_dict={X: X_mdl, Y: y})
                #print('loss: ', current_loss)
                #print('g_2:', sess.run(fetches=g_2,feed_dict={X: X_mdl, Y: y}))
                if use_tb:
                    summary = sess.run(fetches=merged_summary, feed_dict={X: X_mdl, Y: y})
                    train_writer.add_summary(summary,i)
                if not np.isfinite(current_loss) or np.isinf(current_loss) or np.isnan(current_loss):
                    print('loss: ',current_loss)
                #pdb.set_trace()
            sess.run(train_step, feed_dict={X: batch_xs, Y: batch_ys})
        #
        c_sgd = w.eval()
        c_pinv.shape = (len(c_pinv),1)
        #
        print('\n---- Learning params')
        print('Degree_mdl = {}, N = {}, M = {}, eta = {}, nb_iter = {}'.format(Degree_mdl,N,M,eta,nb_iter))
        #
        print('\n---- statistics about learned params')
        print('||c_sgd - c_pinv|| = ', np.linalg.norm(c_sgd - c_pinv,2))
        #
        print('c_sgd.shape: ', c_sgd.shape)
        print('c_pinv.shape: ', c_pinv.shape)
        print('c_rls.shape: ', c_rls.shape)
        print('norm(c_sgd): ', np.linalg.norm(c_sgd))
        print('norm(c_pinv): ', np.linalg.norm(c_pinv))
        print('norm(c_rls): ', np.linalg.norm(c_rls))

        #
        print(' J(c_sgd) = ', sess.run(fetches=loss, feed_dict={X: X_mdl, Y: y}) )
        Xc_pinv = np.dot(X_mdl,c_pinv)
        print( ' J(c_pinv) = ',(1/N)*(np.linalg.norm(y-Xc_pinv)**2) )
        Xc_rls = np.dot(X_mdl,c_rls)
        print( ' J(c_rls) = ',(1/N)*(np.linalg.norm(y-Xc_rls)**2) )
        #
        seconds = (time.time() - start_time)
        print('\a')
        ##
        x_horizontal = np.linspace(lb,ub,1000)
        X_plot = poly_kernel_matrix(x_horizontal,D_sgd-1)
        #plots
        p_sgd, = plt.plot(x_horizontal, np.dot(X_plot,c_sgd))
        p_pinv, = plt.plot(x_horizontal, np.dot(X_plot,c_pinv))
        #p_rls, = plt.plot(x_horizontal, np.dot(X_plot,c_rls))
        p_data, = plt.plot(x_true,y,'ro')
        #
        #p_list=[p_sgd,p_pinv,p_rls,p_data]
        #p_list=[p_data]
        p_list=[p_sgd,p_pinv,p_data]
        #p_list=[p_pinv,p_data]
        #plt.legend(p_list,['sgd curve Degree_mdl='+str(D_sgd-1),'min norm (pinv) Degree_mdl='+str(D_pinv-1),'rls regularization lambda={} Degree_mdl={}'.format(lambda_rls,D_rls-1),'data points'])
        plt.legend(p_list,['sgd curve Degree_mdl='+str(D_sgd-1),'min norm (pinv) Degree_mdl='+str(D_pinv-1),'data points'])
        #plt.legend(p_list,['min norm (pinv) Degree_mdl='+str(D_pinv-1),'data points'])
        plt.ylabel('f(x)')
        plt.show()
        ##
        minutes = seconds/ 60
        hours = minutes/ 60
        print("--- %s seconds ---" % seconds )
        print("--- %s minutes ---" % minutes )
        print("--- %s hours ---" % hours )
        print('\a')

if __name__ == '__main__':
    #tf.app.run()
    main()
    print('\a')
