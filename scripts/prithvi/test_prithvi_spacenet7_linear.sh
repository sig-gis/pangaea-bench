torchrun pangaea/run.py \
    --config-name=train \
    dataset=spacenet7 \
    encoder=prithvi \
    decoder=seg_linear\
    preprocessing=seg_resize_input_layer \
    criterion=dice \
    task=segmentation