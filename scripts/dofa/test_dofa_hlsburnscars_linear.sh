torchrun pangaea/run.py \
    --config-name=train \
    dataset=hlsburnscars \
    encoder=dofa\
    decoder=seg_linear\
    preprocessing=seg_default\
    criterion=cross_entropy \
    task=segmentation