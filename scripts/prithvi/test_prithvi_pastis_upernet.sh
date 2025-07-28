torchrun pangaea/run.py \
 --config-name=train \
 dataset=pastis \
 encoder=prithvi \
 decoder=seg_upernet_mt_ltae \
 preprocessing=seg_resize \
 criterion=dice \
 task=segmentation \
 task.evaluator.inference_mode=whole \
 limited_label_train=0.1 \
 limited_label_strategy=random