torchrun pangaea/run.py \
    --config-name=train \
    dataset=xview2 \
    encoder=terramind_large \
    decoder=seg_siamupernet_conc \
    preprocessing=seg_default \
    criterion=cross_entropy \
    task=change_detection \
    limited_label_train=0.1 \
    limited_label_strategy=stratified