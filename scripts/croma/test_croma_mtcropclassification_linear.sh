torchrun pangaea/run.py \
    --config-name=train \
    dataset=mtcropclassification \
    encoder=croma_optical\
    decoder=seg_linear_mt_ltae\
    preprocessing=seg_default \
    criterion=cross_entropy \
    task=segmentation