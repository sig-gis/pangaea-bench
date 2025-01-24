conda activate pangaea-bench

HYDRA_FULL_ERROR=1 torchrun pangaea/run.py \
    --config-name=train \
    dataset=faofracd \
    encoder=dofa \
    decoder=cls_linear_mt_ltae \
    preprocessing=cls_resize \
    criterion=cross_entropy \
    task=classification