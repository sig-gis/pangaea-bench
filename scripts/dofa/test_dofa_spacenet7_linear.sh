torchrun pangaea/run.py \
    --config-name=train \
    dataset=spacenet7 \
    encoder=dofa \
    decoder=seg_linear\
    preprocessing=seg_irregular_images\
    criterion=dice \
    task=segmentation