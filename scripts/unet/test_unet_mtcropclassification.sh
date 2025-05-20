torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 pangaea/run.py \
    --config-name=train \
   dataset=mtcropclassification\
   encoder=unet_encoder_mi \
   decoder=seg_unet \
   preprocessing=seg_default \
   criterion=cross_entropy \
   task=segmentation \
   finetune=True