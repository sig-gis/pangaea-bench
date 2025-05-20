torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 pangaea/run.py \
    --config-name=train \
    dataset=mados \
    encoder=scalemae\
    decoder=seg_upernet\
    preprocessing=seg_resize_input_layer \
    criterion=cross_entropy \
    task=segmentation