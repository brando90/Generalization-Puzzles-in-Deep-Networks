import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def poly_kernel_matrix( x,D ):
    N = len(x)
    for n in range(N)
        for d in range(D+1)
            Kern[n,d+1] = x[n]**d;
    return Kern

def get_batch(X,Y,M):
    batch_xs = np.random.choice(X,size=M)
    batch_ys = np.random.choice(Y,size=M)
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
    x = linspace(lb,ub,5)
    y = np.array([0,1,0,-1,0]).transpose()
    X_true = poly_kernel_matrix( x,D )
    c_true = linalg.pinv(X_true)*y
    ##
    # N = 5000
    ##
    #l2loss = tf.nn.l2_loss(y - )
    X = tf.placeholder(tf.float32, [None, D_true])
    Y = tf.placeholder(tf.float32, [None, 1])
    w = tf.Variable( tf.zeros([D_sgd,1]) )
    y_ = X*w
    l2loss = tf.reduce_sum(Y - W*c)
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(l2loss)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # Train
        for _ in range(1000):
            batch_xs, batch_ys = get_batch(X,Y,M)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        #
        c_sgd = w.eval()
        print('c_sgd: ',c_sgd)
        print('c_true: 'c_true)
        print(' c_sgd - c_true', np.norm(c_sgd - c_true,2))


if __name__ == '__name__':
    main()
