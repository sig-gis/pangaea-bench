torchrun pangaea/run.py \
    --config-name=train \
    dataset=spacenet7 \
    encoder=croma_optical \
    decoder=seg_linear\
    preprocessing=seg_default\
    criterion=dice \
    task=segmentation