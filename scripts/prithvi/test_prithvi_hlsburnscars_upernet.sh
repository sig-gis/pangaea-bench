torchrun pangaea/run.py \
    --config-name=train \
    dataset=hlsburnscars \
    encoder=prithvi \
    decoder=seg_upernet\
    preprocessing=seg_default \
    criterion=cross_entropy \
    task=segmentation