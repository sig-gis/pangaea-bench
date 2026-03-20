torchrun pangaea/run.py \
    --config-name=train \
    dataset=fuels_reduced_labels \
    encoder=prithvi\
    decoder=seg_upernet_mt_ltae\
    preprocessing=seg_default \
    criterion=cross_entropy \
    task=segmentation