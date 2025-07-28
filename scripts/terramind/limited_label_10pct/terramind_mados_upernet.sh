torchrun pangaea/run.py \
    --config-name=train \
    dataset=mados \
    encoder=terramind_large\
    decoder=seg_upernet\
    preprocessing=seg_default \
    criterion=dice \
    task=segmentation \
    limited_label_train=0.1 \
    limited_label_strategy=stratified