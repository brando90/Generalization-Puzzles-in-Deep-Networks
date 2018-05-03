import os

import time
import numpy as np

import torch

import socket

from email.message import EmailMessage
import smtplib
import os

from pdb import set_trace as st

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

def report_times(start_time,meta_str=''):
    '''
        returns how long has passed since start_time was called.

        start_time is assumed to be some time.time() at some earlier point in the code. Then this function calls its
        own call to current time and just computes a difference
    '''
    ## REPORT TIMES
    seconds = (time.time() - start_time)
    minutes = seconds/ 60
    hours = minutes/ 60
    print(f"--- {seconds} {'seconds '+meta_str} ---")
    print(f"--- {minutes} {'minutes '+meta_str} ---")
    print(f"--- {hours} {'hours '+meta_str} ---")
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

def get_hostname():
    hostname = socket.gethostname()
    if 'polestar-old' in hostname or hostname=='gpu-16' or hostname=='gpu-17':
        return 'polestar-old'
    elif 'openmind' in hostname:
        return 'OM'
    else:
        return hostname
####

def send_email(message,destination):
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    # not a real email account nor password, its all ok!
    server.login('slurm.miranda@gmail.com', 'dummy1234!@#$')

    ## SLURM Job_id=374_* (374) Name=flatness_expts.py Ended, Run time 10:19:54, COMPLETED, ExitCode [0-0]
    msg = EmailMessage()
    msg.set_content(message)

    msg['Subject'] = get_hostname()
    msg['From'] = 'slurm.miranda@gmail.com'
    msg['To'] = destination
    server.send_message(msg)

if __name__ == '__main__':
    send_email('msg','brando90@mit.edu')
