torchrun pangaea/run.py \
    --config-name=train \
   dataset=mtbs\
   encoder=unet_encoder_mi \
   decoder=seg_unet \
   preprocessing=seg_default \
   criterion=cross_entropy \
   task=segmentation \
   finetune=True