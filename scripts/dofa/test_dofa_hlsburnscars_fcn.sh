HYDRA_FULL_ERROR=1 torchrun pangaea/run.py \
    --config-name=train \
    dataset=hlsburnscars \
    encoder=dofa\
    decoder=seg_fcn\
    preprocessing=seg_default\
    criterion=cross_entropy \
    task=segmentation