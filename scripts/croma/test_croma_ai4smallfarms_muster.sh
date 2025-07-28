 torchrun pangaea/run.py \
   --config-name=train \
   dataset=ai4smallfarms \
   encoder=croma_optical \
   decoder=seg_upernet \
   preprocessing=seg_default\
   criterion=dice \
   task=segmentation