torchrun pangaea/run.py \
    --config-name=train \
    dataset=ai4smallfarms \
    encoder=terramind_large\
    decoder=seg_upernet \
    preprocessing=seg_default \
    criterion=dice \
    task=segmentation \
    limited_label_train=0.5 \
    limited_label_strategy=random