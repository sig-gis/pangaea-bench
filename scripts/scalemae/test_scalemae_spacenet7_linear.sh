torchrun pangaea/run.py \
    --config-name=train \
    dataset=spacenet7 \
    encoder=scalemae \
    decoder=seg_linear\
    preprocessing=seg_default\
    criterion=dice \
    task=segmentation