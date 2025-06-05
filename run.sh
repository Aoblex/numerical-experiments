#!/bin/bash

#########
# MNIST #
#########

python mnist.py \
    --task-name 'MNIST' \
    --reg 0.001 \
    --norm l1 \
    --source     2   239 17390 34860 \
    --target 54698 43981 49947 45815 \
    --methods BCD APDAGD LBFGS-Dual Newton SSNS SPLR


#################
# Fashion-MNIST #
#################

python fashion_mnist.py \
    --task-name 'Fashion-MNIST' \
    --reg 0.001 \
    --norm l1 \
    --source     2   239 17390 34860 \
    --target 54698 43981 49947 45815 \
    --methods BCD APDAGD LBFGS-Dual Newton SSNS SPLR

############
# ImageNet #
############

python imagenette.py \
    --task-name 'ImageNet' \
    --reg 0.01 0.001 \
    --norm l1 \
    --source 'tench' \
    --target 'cassette player' \
    --methods BCD APDAGD LBFGS-Dual Newton SSNS SPLR \
    --save-matrices

####################
# synthetic data I #
####################

python synthetic1.py \
    --task-name 'Synthetic I' \
    --reg 0.001 \
    --size 1000 \
    --methods BCD APDAGD LBFGS-Dual Newton SSNS SPLR

python synthetic1.py \
    --task-name 'Synthetic I' \
    --reg 0.001 \
    --size 5000 10000 \
    --methods BCD LBFGS-Dual SSNS SPLR

#####################
# synthetic data II #
#####################

python synthetic2.py \
    --task-name 'Synthetic II' \
    --reg 0.001 \
    --size 1000 \
    --methods BCD APDAGD LBFGS-Dual Newton SSNS SPLR

python synthetic2.py \
    --task-name 'Synthetic II' \
    --reg 0.001 \
    --size 5000 10000 \
    --methods BCD LBFGS-Dual SSNS SPLR

##################
# Ablation study #
##################

python synthetic1.py \
    --task-name 'Synthetic I - Ablation' \
    --reg 0.001 \
    --size 1000 5000 10000 \
    --methods SPLR 'Sparse Newton'

python synthetic2.py \
    --task-name 'Synthetic II - Ablation' \
    --reg 0.001 \
    --size 1000 5000 10000 \
    --methods SPLR 'Sparse Newton'

####################################
# Extra test examples (eta = 0.01) #
####################################

# MNIST
python mnist.py \
    --task-name 'MNIST - Extra' \
    --reg 0.01 \
    --norm l2 \
    --source     2   239 17390 34860 \
    --target 54698 43981 49947 45815 \
    --methods BCD APDAGD LBFGS-Dual Newton SSNS SPLR

# Fashion-MNIST
python fashion_mnist.py \
    --task-name 'FashionMNIST - Extra' \
    --reg 0.01 \
    --norm l2 \
    --source     2   239 17390 34860 \
    --target 54698 43981 49947 45815 \
    --methods BCD APDAGD LBFGS-Dual Newton SSNS SPLR

# ImageNet
python imagenette.py \
    --task-name 'ImageNet - Extra' \
    --reg 0.01 0.001 \
    --norm l1 l2 \
    --source 'tench'           'tench'  'tench'         'tench' \
    --target 'cassette player' 'church' 'garbage truck' 'golf ball' \
    --methods BCD APDAGD LBFGS-Dual Newton SSNS SPLR \
    --save-matrices

# synthetic data I
python synthetic1.py \
    --task-name 'Synthetic I - Extra' \
    --reg 0.01 \
    --size 1000 \
    --methods BCD APDAGD LBFGS-Dual Newton SSNS SPLR

python synthetic1.py \
    --task-name 'Synthetic I - Extra' \
    --reg 0.01 \
    --size 5000 10000 \
    --methods BCD LBFGS-Dual SSNS SPLR

# synthetic data II
python synthetic2.py \
    --task-name 'Synthetic II - Extra' \
    --reg 0.01 \
    --size 1000 \
    --methods BCD APDAGD LBFGS-Dual Newton SSNS SPLR

python synthetic2.py \
    --task-name 'Synthetic II - Extra' \
    --reg 0.01 \
    --size 5000 10000 \
    --methods BCD LBFGS-Dual SSNS SPLR

################################
# Eigenvalue of Sparse Hessian #
################################

python sparse_eigen.py \
    --n 100 \
    --m 100 \
    --stride 50 \
    --repeat 10 