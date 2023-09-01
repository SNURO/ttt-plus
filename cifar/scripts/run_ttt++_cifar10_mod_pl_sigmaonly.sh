#! /usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:$(pwd)

DATASET=cifar10

# ===================================

LEVEL=5

if [ "$#" -lt 2 ]; then
	CORRUPT=snow

	# METHOD=ssl
	# METHOD=align
	METHOD=both
	NSAMPLE=100000
else
	CORRUPT=$1
	METHOD=$2
	NSAMPLE=$3
fi

# ===================================

SCALE_EXT=0.0005
SCALE_SSH=0.0002
LR=0.001
BS_SSL=256
BS_ALIGN=256
QS=1536
NEPOCH=20
DIVERGENCE=coral
DATADIR='/gallery_tate/dongyeon.woo/wonjae/TTTdata/'

echo 'DATASET: '${DATASET}
echo 'CORRUPT: '${CORRUPT}
echo 'METHOD:' ${METHOD}
echo 'DIVERGENCE:' ${DIVERGENCE}
echo 'LR:' ${LR}
echo 'SCALE_EXT:' ${SCALE_EXT}
echo 'SCALE_SSH:' ${SCALE_SSH}
echo 'BS_SSL:' ${BS_SSL}
echo 'NSAMPLE:' ${NSAMPLE}
echo 'NEPOCH:' ${NEPOCH}

# ===================================

printf '\n---------------------\n\n'

#--resume /home2/wonjae.roh/tttplus/ttt-plus/cifar/results/${DATASET}_joint_resnet50 \
#--resume /home2/wonjae.roh/tttplus/simclr/save/cifar10_models/Joint_cifar10_resnet50_lr_1.0_decay_0.0001_bsz_256_temp_0.5_trial_1_balance_0.9_cosine \

python ttt++classwise_final.py \
	--dataroot ${DATADIR} \
	--resume /home2/wonjae.roh/tttplus/ttt-plus/cifar/results/${DATASET}_joint_resnet50 \
	--outf /home2/wonjae.roh/tttplus/ttt-plus/cifar/results/final/sigma_only/${DATASET}_sigma_only_PL_${CORRUPT} \
	--corruption ${CORRUPT} \
	--level ${LEVEL} \
	--workers 36 \
	--fix_ssh \
	--batch_size ${BS_SSL} \
	--batch_size_align ${BS_ALIGN} \
	--queue_size ${QS} \
	--lr ${LR} \
	--scale_ext ${SCALE_EXT} \
	--scale_ssh ${SCALE_SSH} \
	--method ${METHOD} \
	--divergence ${DIVERGENCE} \
	--align_ext \
	--perfect_label \
	--num_sample ${NSAMPLE} \
	--tsne \
	--nepoch ${NEPOCH} \


	#--align_ssh \