def plot_pts():
    fig2 = plt.figure()
    p_grads, = plt.plot([200,375,500], [0.45283,0.1125,0.02702],color='g')
    plt.legend([p_grads],['L2 norm'])
    plt.title('How SGD approaches pseudoinverse')
    plt.xlabel('iterations (thousands)')
    plt.ylabel('L2 norm between SGD solution and pinv')
    plt.show()

def plot_lnorm(p):
    fig2 = plt.figure()
    x_axis = [5,10,50,100,200]
    if p == 1:
        y_axis_l_pinv = [66.51,59.66,70.00,72.91,74.70]
        y_axis_l_sgd = [28.68,19.33,50.03,108.89,183.7]
    else:
        y_axis_l_sgd = [14.28,8.571,10.71,14.43,17.41]
        y_axis_l_pinv = [34.29,23.46,19.57,19.43,19.38]
    p_l_sgd, = plt.plot(x_axis, y_axis_l_sgd,color='g')
    p_l_pinv, = plt.plot(x_axis, y_axis_l_pinv,color='r')
    p_data, = plt.plot(x_axis,y_axis_l_sgd,'go')
    p_data, = plt.plot(x_axis,y_axis_l_pinv,'ro')
    plt.legend([p_l_sgd,p_l_pinv],['L{} norm SGD'.format(p),'L{} norm minimum norm'.format(p)])
    plt.title('SGD vs minimum norm solution L{} norm comparison'.format(p))
    plt.xlabel('Polynomial Degree of model')
    plt.ylabel('L{} norm of parameters'.format(p))
    plt.show()

def plot_generalization():
    fig2 = plt.figure()
    x_axis = [5,10,50,100,200]
    #
    y_axis_l_sgd = [0.03707,0.08483,0.04515,0.09546,1.880]
    y_axis_l_pinv = [0.007742,0.008322,2.604,5.113,7.563]
    #
    p_l_sgd, = plt.plot(x_axis, y_axis_l_sgd,color='g')
    p_l_pinv, = plt.plot(x_axis, y_axis_l_pinv,color='r')
    p_data, = plt.plot(x_axis,y_axis_l_sgd,'go')
    p_data, = plt.plot(x_axis,y_axis_l_pinv,'ro')
    plt.legend([p_l_sgd,p_l_pinv],['Generalization error curve of SGD solution','Generalization error curve of Minimum norm solution'])
    plt.title('SGD vs minimum norm solution generalization error comparison')
    plt.xlabel('Polynomial Degree of model')
    plt.ylabel('Generalization error (with L2 loss)')
    plt.show()
