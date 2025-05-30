#!/bin/bash

####################
# synthetic data I #
####################

python synthetic1.py \
    --reg 0.001 \
    --size 1000 \
    --methods BCD APDAGD LBFGS-Dual Newton SSNS SPLR

python synthetic1.py \
    --reg 0.001 \
    --size 5000 10000 \
    --methods BCD LBFGS-Dual SSNS SPLR


#####################
# synthetic data II #
#####################

python synthetic2.py \
    --reg 0.001 \
    --size 1000 \
    --methods BCD APDAGD LBFGS-Dual Newton SSNS SPLR

python synthetic2.py \
    --reg 0.001 \
    --size 5000 10000 \
    --methods BCD LBFGS-Dual SSNS SPLR


#########
# MNIST #
#########

python mnist.py \
    --reg 0.001 \
    --norm l1 \
    --source     2   239 17390 34860 \
    --target 54698 43981 49947 45815 \
    --methods BCD APDAGD LBFGS-Dual Newton SSNS SPLR


#################
# Fashion-MNIST #
#################

python fashion_mnist.py \
    --reg 0.001 \
    --norm l1 \
    --source     2   239 17390 34860 \
    --target 54698 43981 49947 45815 \
    --methods BCD APDAGD LBFGS-Dual Newton SSNS SPLR

############
# ImageNet #
############

python imagenette.py \
    --reg 0.01 0.001 \
    --norm l1 l2 \
    --source 'tench' \
    --target 'cassette player' \
    --methods BCD APDAGD LBFGS-Dual Newton SSNS SPLR


##################
# Ablation study #
##################

python synthetic1.py \
    --reg 0.001 \
    --size 1000 5000 10000 \
    --methods SPLR 'Sparse Newton' \
    --force-rerun

python synthetic2.py \
    --reg 0.001 \
    --size 1000 5000 10000 \
    --methods SPLR 'Sparse Newton' \
    --force-rerun

####################################
# Extra test examples (eta = 0.01) #
####################################

# synthetic data I
python synthetic1.py \
    --reg 0.01 \
    --size 1000 \
    --methods BCD APDAGD LBFGS-Dual Newton SSNS SPLR

python synthetic1.py \
    --reg 0.01 \
    --size 5000 10000 \
    --methods BCD LBFGS-Dual SSNS SPLR

# synthetic data II
python synthetic2.py \
    --reg 0.01 \
    --size 1000 \
    --methods BCD APDAGD LBFGS-Dual Newton SSNS SPLR

python synthetic2.py \
    --reg 0.01 \
    --size 5000 10000 \
    --methods BCD LBFGS-Dual SSNS SPLR

# MNIST
python mnist.py \
    --reg 0.01 \
    --norm l1 \
    --source     2   239 17390 34860 \
    --target 54698 43981 49947 45815 \
    --methods BCD APDAGD LBFGS-Dual Newton SSNS SPLR

# Fashion-MNIST
python fashion_mnist.py \
    --reg 0.01 \
    --norm l1 \
    --source     2   239 17390 34860 \
    --target 54698 43981 49947 45815 \
    --methods BCD APDAGD LBFGS-Dual Newton SSNS SPLR

# ImageNet
python imagenette.py \
    --reg 0.01 0.001 \
    --norm l1 l2 \
    --source 'tench'           'tench'  'tench'         'tench' \
    --target 'cassette player' 'church' 'garbage truck' 'golf ball' \
    --methods BCD APDAGD LBFGS-Dual Newton SSNS SPLR