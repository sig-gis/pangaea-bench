torchrun pangaea/run.py \
   --config-name=train \
   dataset=mados \
   encoder=unet_encoder \
   decoder=seg_unet \
   preprocessing=seg_resize_input_layer \
   criterion=cross_entropy \
   task=segmentation \
   finetune=True