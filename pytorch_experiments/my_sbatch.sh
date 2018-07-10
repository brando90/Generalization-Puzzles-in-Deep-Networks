#!/bin/bash

for i in {0..5};
do
    export CUDA_VISIBLE_DEVICES=$i
    nohup python flatness_expts.py -mdl GBoixNet -label_corrupt_prob 0.5 -train_alg SGD -epochs 300 -exptlabel MoreRLInits &
    #nohup python flatness_expts.py -mdl GBoixNet -train_alg SGD -epochs 3200 -exptlabel SGD_ManyRuns_Momentum0.9_3200 &
    #nohup python flatness_expts.py -train_alg brando_chiyuan_radius_inter -epochs 20 -mdl radius_flatness -nb_dirs 1000 -net_name NL -exptlabel RadiusFlatnessNL_samples20_RLarge50 &
    #nohup python flatness_expts.py  -train_alg brando_chiyuan_radius_inter -epochs 20 -mdl radius_flatness -nb_dirs 500 -net_name NL -r_large 12 -exptlabel RadiusFlatnessNL_samples20 &
    sleep 1
done

for i in {3..5};
do
    export CUDA_VISIBLE_DEVICES=$i
    #nohup python flatness_expts.py -train_alg brando_chiyuan_radius_inter -epochs 20 -mdl radius_flatness -nb_dirs 1000 -net_name RLNL -exptlabel RadiusFlatnessRLNL_samples20_RLarge50 &
    #nohup python flatness_expts.py -train_alg brando_chiyuan_radius_inter -epochs 20 -mdl radius_flatness -nb_dirs 500 -net_name RLNL -r_large 12 -exptlabel RadiusFlatnessRLNL_samples20 &
    sleep 1
done

echo 'sbatch submission DONE'