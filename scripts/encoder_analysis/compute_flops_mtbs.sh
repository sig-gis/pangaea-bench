#!/bin/bash

  
python3 compute_flops.py dataset=mtbs    encoder=terramind_large   preprocessing=seg_resize    task=segmentation

python3 compute_flops.py dataset=mtbs    encoder=croma_optical    preprocessing=seg_resize_input_layer    task=segmentation

python3 compute_flops.py dataset=mtbs    encoder=dofa    preprocessing=seg_resize_input_layer    task=segmentation

python3 compute_flops.py dataset=mtbs    encoder=gfmswin  preprocessing=seg_resize  task=segmentation

python3 compute_flops.py dataset=mtbs    encoder=prithvi  preprocessing=seg_resize_input_layer    task=segmentation

python3 compute_flops.py dataset=mtbs    encoder=satlasnet_mi  preprocessing=seg_resize_input_layer    task=segmentation

python3 compute_flops.py dataset=mtbs    encoder=scalemae  preprocessing=seg_resize_input_layer    task=segmentation

python3 compute_flops.py dataset=mtbs    encoder=spectralgpt preprocessing=seg_resize   task=segmentation

python3 compute_flops.py dataset=mtbs    encoder=ssl4eo_data2vec preprocessing=seg_resize   task=segmentation
python3 compute_flops.py dataset=mtbs    encoder=ssl4eo_dino preprocessing=seg_resize   task=segmentation
python3 compute_flops.py dataset=mtbs    encoder=ssl4eo_mae_optical preprocessing=seg_resize  task=segmentation
python3 compute_flops.py dataset=mtbs    encoder=ssl4eo_moco preprocessing=seg_resize    task=segmentation

python3 compute_flops.py dataset=mtbs    encoder=unet_encoder_mi preprocessing=seg_resize_input_layer    task=segmentation

python3 compute_flops.py dataset=mtbs    encoder=resnet50_scratch preprocessing=seg_resize_input_layer    task=segmentation

python3 compute_flops.py dataset=mtbs    encoder=resnet50_pretrained preprocessing=seg_resize_input_layer    task=segmentation

python3 compute_flops.py dataset=mtbs    encoder=vit_scratch preprocessing=seg_resize   task=segmentation

python3 compute_flops.py dataset=mtbs    encoder=vit_mi preprocessing=seg_resize   task=segmentation


