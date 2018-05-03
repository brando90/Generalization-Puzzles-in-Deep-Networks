#!/bin/bash

for i in {0..2};
do
    export CUDA_VISIBLE_DEVICES=$i
    nohup python flatness_expts.py -cuda -train_alg brando_chiyuan_radius_inter -epochs 20 -mdl radius_flatness -nb_dirs 1000 -net_name NL -exptlabel RadiusFlatnessNL_samples20_RLarge50 &
    sleep 1
done

for i in {3..5};
do
    export CUDA_VISIBLE_DEVICES=$i
    nohup python flatness_expts.py -cuda -train_alg brando_chiyuan_radius_inter -epochs 20 -mdl radius_flatness -nb_dirs 1000 -net_name RLNL -exptlabel RadiusFlatnessRLNL_samples20_RLarge50 &
    sleep 1
done

echo 'sbatch submission DONE'