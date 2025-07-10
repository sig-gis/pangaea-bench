torchrun pangaea/run.py \
    --config-name=train \
    dataset=hlsburnscars \
    encoder=terramind_large \
    decoder=seg_upernet \
    preprocessing=seg_default \
    criterion=cross_entropy \
    task=segmentation