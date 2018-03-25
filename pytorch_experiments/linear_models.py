

def get_hermite_coeffs(X,Y):
    # TODO
    C = np.polynomial.hermite.hermfit(X,Y)
    return C


class Trainer:
    '''
    Traings with given alg and collects stats
    '''

    def __init__(self,nb_mdls,train_alg,stats_collector):
        '''
        nb_mdls =
        train_alg = training algoritm for models
        stats_collector = function that collects the stats
        '''
        #TODO, need to rethinking how the indexing of each model complexity works as training proceeds

    def train_mdls(self):
        '''
        trains all the models
        '''
    # TODO

    def get_stats_dict(self):
        '''
        returns NamedDict of stats
        '''
    # TODO
