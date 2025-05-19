torchrun pangaea/run.py \
    --config-name=train \
    dataset=sen1floods11 \
    encoder=croma_joint \
    decoder=seg_upernet \
    preprocessing=seg_resize_input_layer \
    criterion=cross_entropy \
    task=segmentation