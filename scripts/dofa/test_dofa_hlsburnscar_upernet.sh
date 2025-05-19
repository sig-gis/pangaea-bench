HYDRA_FULL_ERROR=1 torchrun pangaea/run.py \
    --config-name=train \
    dataset=hlsburnscars \
    encoder=dofa\
    decoder=seg_upernet\
    preprocessing=seg_resize_input_layer \
    criterion=cross_entropy \
    task=segmentation