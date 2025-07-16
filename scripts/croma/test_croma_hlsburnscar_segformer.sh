torchrun pangaea/run.py \
    --config-name=train \
    dataset=hlsburnscars \
    encoder=croma_optical\
    decoder=seg_segformer\
    preprocessing=seg_default\
    criterion=cross_entropy \
    task=segmentation