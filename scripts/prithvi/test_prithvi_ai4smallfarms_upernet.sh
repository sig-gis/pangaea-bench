 torchrun pangaea/run.py \
   --config-name=train \
   dataset=ai4smallfarms \
   encoder=prithvi\
   decoder=seg_upernet \
   preprocessing=seg_irregular_images \
   criterion=cross_entropy \
   task=segmentation