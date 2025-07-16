torchrun pangaea/run.py \
    --config-name=train \
    dataset=hlsburnscars \
    encoder=scalemae \
    decoder=seg_linear\
    preprocessing=seg_linear\
    criterion=cross_entropy \
    task=segmentation