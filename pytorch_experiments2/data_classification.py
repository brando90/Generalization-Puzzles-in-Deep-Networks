import numpy as np

def get_quadratic_plane_classification_data_set(D0,lb,ub, N_train,N_test):
    '''
    data set with feature space phi(x)=[x0,x1,x2]=[1,x,x^2]
    separating hyperplane = [0,1,-2]
    corresponds to line x2 = 0.5x1
    '''
    ## target function
    freq_sin = 4
    #f_target = lambda x: np.sin(2*np.pi*freq_sin*x)
    #f_target = lambda x: (x-0.25)*(x-0.75)*(x+0.25)*(x+0.75)
    def f_target(x):
        poly_feat = PolynomialFeatures(degree=2)
        x_feature = poly_feat.fit_transform(x) # N x D, [1, x, x^2]
        normal = np.zeros((1,x_feature.shape[1])) # 1 x D
        normal[:,[0,1,2]] = [0,1,-2]
        score = np.dot(normal,x_feature.T)
        label = score > 0
        return label.astype(int)
    ## define x
    X_train = np.linspace(lb,ub,N_train).reshape((N_train,D0))
    X_test = np.linspace(lb,ub,N_test).reshape((N_test,D0))
    ## get y's
    Y_train = f_target(X_train)
    Y_test = f_target(X_test)
    return X_train,X_test, Y_train,Y_test


def get_2D_classification_data():
    '''
    Returns x in R^2 classification data
    '''
    # TODO
    return
