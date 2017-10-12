#!/usr/bin/env bash

#SBATCH --mem=7000
#SBATCH --time=7-00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=brando90@mit.edu

py_path=~/home_simulation_research/overparametrized_experiments/pytorch_experiments
cd $py_path
## UNIT Test
SLURM_JOBID=2
#python bulk_experiment_dispatcher.py -expt_type lambda -lb 10 -ub 20 -num 2 -num_rep 2 -save True -sj $SLURM_JOBID -rt_wp VW
python bulk_experiment_dispatcher.py -expt_type iterations -lb 1000 -ub 2000 -num 2 -num_rep 2 -save True -sj $SLURM_JOBID -rt_wp VW
## LAMBDA
#python bulk_experiment_dispatcher.py -expt_type lambda -lb 50 -ub 10000 -num 50 -num_rep 15 -save True -sj $SLURM_JOBID -rt_wp VW
## ITERATIONS
#python bulk_experiment_dispatcher.py -expt_type iterations -lb 10000 -ub 60000 -num 50 -num_rep 15 -save True -sj $SLURM_JOBID -rt_wp VW
