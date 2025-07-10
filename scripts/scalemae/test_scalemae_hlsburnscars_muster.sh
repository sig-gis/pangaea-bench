torchrun pangaea/run.py \
    --config-name=train \
    dataset=hlsburnscars \
    encoder=scalemae\
    decoder=seg_muster\
    preprocessing=seg_default \
    criterion=cross_entropy \
    task=segmentation