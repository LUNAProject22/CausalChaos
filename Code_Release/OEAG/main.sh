#########################################################################
# We build upon the following work.
# Original file credits as follows. We thank the authors of this work and
# all the related work.
# If you find this work useful, please consider citing the following and 
# related work.
# File Name: main.sh
# Author: Xiao Junbin
# mail: junbin@comp.nus.edu.sg
# @InProceedings{xiao2021next,
#     author    = {Xiao, Junbin and Shang, Xindi and Yao, Angela and Chua, Tat-Seng},
#     title     = {NExT-QA: Next Phase of Question-Answering to Explaining Temporal Actions},
#     booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
#     month     = {June},
#     year      = {2021},
#     pages     = {9777-9786}
# }
# @article{causalchaos2024,
#   title={CausalChaos! Dataset for Comprehensive Causal Action Question Answering Over Longer Causal Chains Grounded in Dynamic Visual Scenes},
#   author={Parmar, Paritosh and Peh, Eric and Chen, Ruirui and Lam, Ting En and Chen, Yuhan and Tan, Elston and Fernando, Basura},
#   journal={arXiv preprint arXiv:2404.01299},
#   year={2024}
# }
#########################################################################
#!/bin/bash
GPU=$1
MODE=$2
DATASET=$3 
VERSION=$4 
SPLIT=$5

CUDA_VISIBLE_DEVICES=$GPU python main_qa.py \
	--mode $MODE --dataset $DATASET --version $VERSION --split $SPLIT