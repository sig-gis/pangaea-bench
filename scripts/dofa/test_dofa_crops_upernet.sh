HYDRA_FULL_ERROR=1 torchrun pangaea/run.py \
    --config-name=train \
    dataset=paired_modality_crops_l8 \
    encoder=dofa\
    decoder=seg_fcn\
    preprocessing=seg_default \
    criterion=cross_entropy \
    task=segmentation