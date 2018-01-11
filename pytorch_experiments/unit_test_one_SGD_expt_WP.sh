export SLURM_JOBID=2
for i in {25..25}; do
  export SLURM_ARRAY_TASK_ID=$i
  python one_SGD_expt_WP.py
  #python one_SGD_expt_WP.py &
done
