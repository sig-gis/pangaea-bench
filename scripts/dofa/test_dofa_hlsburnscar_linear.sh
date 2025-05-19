torchrun pangaea/run.py \
    --config-name=train \
    dataset=hlsburnscars \
    encoder=dofa\
    decoder=seg_linear\
    preprocessing=seg_resize_input_layer \
    criterion=cross_entropy \
    task=segmentation