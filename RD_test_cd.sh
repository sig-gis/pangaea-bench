conda activate pangaea-bench

torchrun pangaea/run.py \
    --config-name=train \
    dataset=xview2 \
    encoder=prithvi \
    decoder=seg_siamupernet_conc \
    preprocessing=seg_default \
    criterion=cross_entropy \
    task=change_detection