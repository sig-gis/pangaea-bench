conda activate pangaea-bench

torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
 --config-name=train \
 dataset=hlsburnscars \
 encoder=dofa \
 decoder=seg_upernet\
 preprocessing=seg_default \
 criterion=cross_entropy \
 task=segmentation