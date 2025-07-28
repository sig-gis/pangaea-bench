torchrun pangaea/run.py \
    --config-name=train \
    dataset=biomasters \
    encoder=terramind_large_mm \
    decoder=reg_upernet_mt_ltae \
    preprocessiong=reg_default \
    criterion=mse \
    task=regression \
    limited_label_train=0.1 \
    limited_label_strategy=stratified
