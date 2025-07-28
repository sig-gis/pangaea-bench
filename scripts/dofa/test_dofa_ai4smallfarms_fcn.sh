 torchrun pangaea/run.py \
   --config-name=train \
   dataset=ai4smallfarms \
   encoder=dofa \
   decoder=seg_fcn \
   preprocessing=seg_default \
   criterion=cross_entropy \
   task=segmentation