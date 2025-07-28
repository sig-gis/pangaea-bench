torchrun pangaea/run.py \
    --config-name=train \
    dataset=sen1floods11 \
    encoder=terramind_large_mm\
    decoder=seg_upernet\
    preprocessing=seg_default \
    criterion=cross_entropy \
    task=segmentation \
    limited_label_train=0.1 \
    limited_label_strategy=stratified
    