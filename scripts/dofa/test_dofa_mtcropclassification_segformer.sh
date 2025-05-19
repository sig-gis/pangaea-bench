torchrun pangaea/run.py \
    --config-name=train \
    dataset=mtcropclassification \
    encoder=dofa\
    decoder=seg_segformer_mt_ltae\
    preprocessing=seg_resize_input_layer \
    criterion=cross_entropy \
    task=segmentation