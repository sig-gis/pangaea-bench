conda activate pangaea-bench

torchrun pangaea/run.py \
    --config-name=train \
    dataset=pastis \
    encoder=dofa \
    decoder=seg_linear_mt_ltae \
    task=segmentation \
    preprocessing=seg_resize \
    criterion=cross_entropy \
    task.evaluator.inference_mode=whole