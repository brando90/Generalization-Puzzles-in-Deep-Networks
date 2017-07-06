import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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

def main():
    ##
    lb, ub = 0,1
    N = 5
    Degree_true= 4
    D_true = Degree_true+1
    D_sgd = Degree_true+1
    D_mdl = Degree_true
    B=1000
    nb_iter = 400000
    ##
    x = np.linspace(lb,ub,5)
    y = np.array([0,1,0,-1,0]).transpose()
    X_true = poly_kernel_matrix( x,Degree_true )
    c_true = np.linalg.pinv(X_true)*y
    ##
    # N = 5000
    ##
    #l2loss = tf.nn.l2_loss(y - )
    X = tf.placeholder(tf.float32, [None, D_true])
    Y = tf.placeholder(tf.float32, [None])
    w = tf.Variable( tf.zeros([D_sgd,1]) )
    f = X*w
    l2loss = tf.reduce_sum(Y - f)
    #
    M = 5
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(l2loss)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # Train
        for _ in range(1000):
            batch_xs, batch_ys = get_batch(X_true,y,M)
            sess.run(train_step, feed_dict={X: batch_xs, Y: batch_ys})
        #
        c_sgd = w.eval()
        print('c_sgd: ',c_sgd)
        print('c_true: ',c_true)
        print(' c_sgd - c_true', np.norm(c_sgd - c_true,2))


if __name__ == '__main__':
    print('__main__')
    main()
    print('end')
