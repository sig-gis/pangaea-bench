torchrun pangaea/run.py \
    --config-name=train \
    dataset=hlsburnscars \
    encoder=scalemae \
    decoder=seg_upernet\
    preprocessing=seg_resize_input_layer \
    criterion=cross_entropy \
    task=segmentation