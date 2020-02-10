
Generalization-Puzzles-in-Deep-Networks
--

In this repository we have the code for experiments produced in the following paper:

A surprising linear relationship predicts test performance in deep networks (https://arxiv.org/pdf/1807.09659.pdf)

How to reproduce
--

To see examples of how to reproduce the experimental results you can see the submission file to slurm written here:

https://github.com/brando90/Generalization-Puzzles-in-Deep-Networks/blob/master/pytorch_experiments/my_sbatch.sh

but an example command is:

```
python flatness_expts.py -mdl GBoixNet -label_corrupt_prob 0.5 -train_alg SGD -epochs 300 -exptlabel MoreRLInits
```

note that the important file is `flatness_expts.py`. The experiments are simple and described in the paper. It should be simple to change the arguments of the script to change the experiment. 

Note that the file:

https://github.com/brando90/Generalization-Puzzles-in-Deep-Networks/blob/master/pytorch_experiments/flatness_expts.py

is a slurm submission script and one can use the array command to send multiple runs of similar experiments.
