torchrun pangaea/run.py \
    --config-name=train \
    dataset=spacenet7 \
    encoder=scalemae \
    decoder=seg_upernet\
    preprocessing=seg_resize_input_layer \
    criterion=dice \
    task=segmentation