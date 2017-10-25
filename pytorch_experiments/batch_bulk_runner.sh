#!/usr/bin/env bash

#SBATCH --mem=7000
#SBATCH --time=7-00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=brando90@mit.edu
#SBATCH --qos==cbmm

py_path=~/home_simulation_research/overparametrized_experiments/pytorch_experiments
echo '$CWD'
echo $CWD
echo '$PWD'
echo $PWD
pwd
echo 'BEFORE CD'
cd $py_path
pwd
echo '$CWD'
echo $CWD
echo '$PWD'
echo $PWD
## UNIT Test
#SLURM_JOBID=2
#python bulk_experiment_dispatcher.py -expt_type lambda -lb 10 -ub 20 -num 2 -num_rep 2 -save True -sj $SLURM_JOBID -rt_wp VW
#python bulk_experiment_dispatcher.py -expt_type iterations -lb 1000 -ub 2000 -num 2 -num_rep 2 -save True -sj $SLURM_JOBID -rt_wp VW
## LAMBDA
#python bulk_experiment_dispatcher.py -expt_type lambda -lb 50 -ub 100000 -num 30 -num_rep 10 -save True -sj $SLURM_JOBID -rt_wp VW
## ITERATIONS
python bulk_experiment_dispatcher.py -expt_type iterations -lb 300000 -ub 5000000 -num 20 -num_rep 5 -save True -sj $SLURM_JOBID -rt_wp VW
