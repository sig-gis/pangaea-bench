torchrun pangaea/run.py \
    --config-name=train \
    dataset=sen1floods11 \
    encoder=scalemae \
    decoder=seg_linear\
    preprocessing=seg_linear \
    criterion=cross_entropy \
    task=segmentation