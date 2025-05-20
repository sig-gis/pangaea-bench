torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 pangaea/run.py \
   --config-name=train \
   dataset=ai4smallfarms \
   encoder=croma_optical \
   decoder=seg_upernet \
   preprocessing=seg_irregular_images \
   criterion=cross_entropy \
   task=segmentation