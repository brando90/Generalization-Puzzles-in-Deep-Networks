import os

import time
import np

import torch

def is_NaN(value):
    '''
    Checks is value is problematic by checking if the value:
    is not finite, is infinite or is already NaN
    '''
    return not np.isfinite(value) or np.isinf(value) or np.isnan(value)

##

def make_and_check_dir(path):
    '''
        tries to make dir/file, if it exists already does nothing else creates it.
    '''
    try:
        os.makedirs(path)
    except OSError:
        pass

def report_times(start_time):
    ## REPORT TIMES
    seconds = (time.time() - start_time)
    minutes = seconds/ 60
    hours = minutes/ 60
    print("--- %s seconds ---" % seconds )
    print("--- %s minutes ---" % minutes )
    print("--- %s hours ---" % hours )
    print('\a')
    return seconds, minutes, hours

####

def save_pytorch_mdl(path_to_save,net):
    ##http://pytorch.org/docs/master/notes/serialization.html
    ##The first (recommended) saves and loads only the model parameters:
    torch.save(net.state_dict(), path_to_save)

def restore_mdl(path_to_save,mdl_class):
    # TODO
    the_model = TheModelClass(*args, **kwargs)
    the_model.load_state_dict(torch.load(PATH))

def save_entire_mdl(path_to_save,the_model):
    torch.save(the_model, path_to_save)

def restore_entire_mdl(path_to_restore):
    '''
    NOTE: However in this case, the serialized data is bound to the specific
    classes and the exact directory structure used,
    so it can break in various ways when used in other projects, or after some serious refactors.
    '''
    the_model = torch.load(path_to_restore)
    return the_model
