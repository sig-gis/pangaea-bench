torchrun pangaea/run.py \
    --config-name=train \
    dataset=croptypemapping \
    encoder=dofa \
    decoder=seg_upernet_mt_ltae \
    preprocessing=seg_resize_input_layer \
    criterion=cross_entropy \
    task=segmentation \
    task.evaluator.inference_mode=whole 