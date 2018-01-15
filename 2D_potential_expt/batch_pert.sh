#!/bin/bash
#SBATCH --mem=7000
#SBATCH --time=0-11:00
#SBATCH --mail-type=END
#SBATCH --mail-user=brando90@mit.edu
#SBATCH --array=1-30

echo start

#alias matlab='/Applications/MATLAB_R2017a.app/bin/matlab -nodesktop -nosplash'
#jid=0
#satid=3

jid=${SLURM_JOBID}
satid=${SLURM_ARRAY_TASK_ID}
print_hist=0
matlab -nodesktop -nosplash -nojvm -r "run_GDL_wedge_perturbations($jid,$satid,$print_hist)"