#!/usr/bin/env python
#SBATCH --mem=7000
#SBATCH --time=7-00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=brando90@mit.edu

import numpy as np
import os

from subprocess import call

## config params
N_lambdas = 3
lb,ub=1,3
lambdas = np.linspace(lb,ub,N_lambdas)
repetitions = len(lambdas)*[2]

##
nb_iterations = 10000

## dispatch jobs
for i in range(len(lambdas)):
    reg_lambda = lambdas[i]
    N_repetitions = repetitions[i]
    #cmd(['sbatch','-a',N_repetitions, '-D', cwd = os.getcwd(), 'python', '--reg_lambda',reg_lambda,'single_sgd_train.py'])
    call(['echo',str(i)])
    call(['sleep',str(3),'&'])
