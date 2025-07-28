torchrun pangaea/run.py \
   --config-name=train \
   dataset=ai4smallfarms \
   encoder=dofa \
   decoder=seg_muster \
   preprocessing=seg_irregular_images \
   criterion=cross_entropy \
   task=segmentation