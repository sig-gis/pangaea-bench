torchrun pangaea/run.py \
    --config-name=train \
    dataset=hlsburnscars \
    encoder=croma_optical\
    decoder=seg_upernet\
    preprocessing=seg_resize_input_layer \
    criterion=cross_entropy \
    task=segmentation