 torchrun pangaea/run.py \
   --config-name=train \
   dataset=ai4smallfarms \
   encoder=scalemae \
   decoder=seg_linear \
   preprocessing=seg_default \
   criterion=dice \
   task=segmentation