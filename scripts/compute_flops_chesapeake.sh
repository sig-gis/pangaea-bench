#!/bin/bash

  
python3 compute_flops.py dataset=mchesapeake-landcover    encoder=terramind_large   preprocessing=seg_resize    task=segmentation

python3 compute_flops.py dataset=mchesapeake-landcover    encoder=croma_optical    preprocessing=seg_resize_input_layer    task=segmentation

python3 compute_flops.py dataset=mchesapeake-landcover    encoder=dofa    preprocessing=seg_resize_input_layer    task=segmentation

python3 compute_flops.py dataset=mchesapeake-landcover    encoder=gfmswin  preprocessing=seg_resize  task=segmentation

python3 compute_flops.py dataset=mchesapeake-landcover    encoder=prithvi  preprocessing=seg_resize_input_layer    task=segmentation

python3 compute_flops.py dataset=mchesapeake-landcover    encoder=satlasnet_si  preprocessing=seg_resize_input_layer    task=segmentation

python3 compute_flops.py dataset=mchesapeake-landcover    encoder=scalemae  preprocessing=seg_resize_input_layer    task=segmentation

python3 compute_flops.py dataset=mchesapeake-landcover    encoder=spectralgpt preprocessing=seg_resize   task=segmentation

python3 compute_flops.py dataset=mchesapeake-landcover    encoder=ssl4eo_data2vec preprocessing=seg_resize   task=segmentation
python3 compute_flops.py dataset=mchesapeake-landcover    encoder=ssl4eo_dino preprocessing=seg_resize   task=segmentation
python3 compute_flops.py dataset=mchesapeake-landcover    encoder=ssl4eo_mae_optical preprocessing=seg_resize  task=segmentation
python3 compute_flops.py dataset=mchesapeake-landcover    encoder=ssl4eo_moco preprocessing=seg_resize    task=segmentation

python3 compute_flops.py dataset=mchesapeake-landcover    encoder=unet_encoder preprocessing=seg_resize_input_layer    task=segmentation

python3 compute_flops.py dataset=mchesapeake-landcover    encoder=resnet50_scratch preprocessing=seg_resize_input_layer    task=segmentation

python3 compute_flops.py dataset=mchesapeake-landcover    encoder=resnet50_pretrained preprocessing=seg_resize_input_layer    task=segmentation

python3 compute_flops.py dataset=mchesapeake-landcover    encoder=vit_scratch preprocessing=seg_resize   task=segmentation

python3 compute_flops.py dataset=mchesapeake-landcover    encoder=vit preprocessing=seg_resize   task=segmentation


