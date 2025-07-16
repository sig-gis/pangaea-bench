torchrun pangaea/run.py \
   --config-name=train \
   dataset=ai4smallfarms \
   encoder=prithvi \
   decoder=seg_linear \
   preprocessing=seg_default \
   criterion=cross_entropy \
   task=segmentation