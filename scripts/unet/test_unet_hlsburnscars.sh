torchrun pangaea/run.py \
    --config-name=train \
   dataset=hlsburnscars \
   encoder=unet_encoder \
   decoder=seg_unet \
   preprocessing=seg_default \
   criterion=cross_entropy \
   task=segmentation \
   finetune=True