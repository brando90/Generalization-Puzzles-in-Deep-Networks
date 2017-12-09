#!/usr/bin/env python
#SBATCH --mem=7000
#SBATCH --time=0-01:00
#SBATCH --qos=cbmm

import os

print('I AM IN PYTHON')
print(os.getcwd())
