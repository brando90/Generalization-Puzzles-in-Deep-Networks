from pdb import set_trace as st

from maps import NamedDict

import utils

class StatsCollector:
    '''
    Class that has all the stats collected during training.
    '''
    def __init__(self,net):
        ''' loss & errors lists'''
        self.train_losses, self.val_losses, self.test_losses = [], [], []
        self.train_errors, self.val_errors, self.test_errors = [], [], []
        self.train_accs, self.val_accs, self.test_accs = [], [], []
        ''' stats related to parameters'''
        nb_param_groups = len( list(net.parameters()) )
        self.grads = [ [] for i in range(nb_param_groups) ]
        self.w_norms = [ [] for i in range(nb_param_groups) ]
        self.perturbations_norms = [ [] for i in range(nb_param_groups) ]
        ''' reference net errors '''
        self.ref_train_losses, self.ref_val_losses, self.ref_test_losses = -1, -1, -1
        self.ref_train_errors, self.ref_val_errors, self.ref_test_errors = -1, -1, -1
        self.ref_train_accs, self.ref_val_accs, self.ref_test_accs = -1, -1, -1

    def collect_mdl_params_stats(self,mdl):
        ''' log parameter stats'''
        for index, W in enumerate(mdl.parameters()):
            self.w_norms[index].append( W.data.norm(2) )
            if W.grad is not None:
                self.grads[index].append( W.grad.data.norm(2) )
                if utils.is_NaN(W.grad.data.norm(2)):
                    raise ValueError(f'Nan Detected error happened at: i={i} loss_val={loss_val}, loss={loss}')

    def append_losses_errors_accs(self,train_loss, train_error, test_loss, test_error):
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
        self.train_errors.append(train_error)
        self.test_errors.append(test_error)
        self.train_accs.append(1.0-train_error)
        self.test_accs.append(1.0-test_error)

    def add_perturbation_norms_from_perturbations(self,net,perturbations):
        for index, W in enumerate(mdl.parameters()):
            self.perturbations_norms[index].append( perturbations[index].norm(2) )

    def record_errors_loss_reference_net(self,criterion,error_criterion,net,trainloader,testloader,enable_cuda):
        train_loss, train_error = evalaute_mdl_data_set(criterion,error_criterion,net,trainloader,enable_cuda)
        test_loss, test_error = evalaute_mdl_data_set(criterion,error_criterion,net,testloader,enable_cuda)
        self.ref_train_losses, self.ref_test_losses = train_loss, train_error
        self.ref_train_accs, self.ref_test_accs = test_loss, test_error

    def get_stats_dict(self):
        stats = NamedDict(
            train_losses=self.train_losses,val_losses=self.val_losses,test_losses=self.test_losses,
            train_errors=self.train_errors,val_errors=self.val_errors,test_errors=self.test_errors,
            train_accs=self.train_accs,val_accs=self.val_accs,test_accs=self.test_accs,
            grads=self.grads,
            w_norms=self.w_norms,
            perturbations=self.perturbations,
            ref_train_losses=self.ref_train_losses,ref_val_losses=self.ref_val_losses,ref_test_losses=self.ref_test_losses,
            ref_train_errors=self.ref_train_errors, ref_val_errors=self.ref_val_errors,ref_test_errors=self.ref_test_errors
            ref_train_accs=self.ref_train_accs,ref_val_accs=self.ref_val_accs,ref_test_accs=self.ref_test_accs
        )
        return stats
