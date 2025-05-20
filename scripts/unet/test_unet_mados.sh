torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 pangaea/run.py \
   --config-name=train \
   dataset=mados \
   encoder=unet_encoder \
   decoder=seg_unet \
   preprocessing=seg_resize_input_layer \
   criterion=cross_entropy \
   task=segmentation \
   finetune=True