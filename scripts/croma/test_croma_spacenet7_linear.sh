torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 pangaea/run.py \
    --config-name=train \
    dataset=spacenet7 \
    encoder=dofa \
    decoder=seg_linear\
    preprocessing=seg_resize_input_layer \
    criterion=dice \
    task=segmentation