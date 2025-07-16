#!/bin/bash

  
python3 embed.py dataset=hlsburnscars    encoder=terramind_large   preprocessing=seg_resize    task=segmentation

python3 embed.py dataset=hlsburnscars    encoder=croma_optical    preprocessing=seg_resize_input_layer    task=segmentation

python3 embed.py dataset=hlsburnscars    encoder=dofa    preprocessing=seg_resize_input_layer    task=segmentation

python3 embed.py dataset=hlsburnscars    encoder=gfmswin  preprocessing=seg_resize  task=segmentation

python3 embed.py dataset=hlsburnscars    encoder=prithvi  preprocessing=seg_resize_input_layer    task=segmentation

python3 embed.py dataset=hlsburnscars    encoder=satlasnet_si  preprocessing=seg_resize_input_layer    task=segmentation

python3 embed.py dataset=hlsburnscars    encoder=scalemae  preprocessing=seg_resize_input_layer    task=segmentation

python3 embed.py dataset=hlsburnscars    encoder=spectralgpt preprocessing=seg_resize   task=segmentation

python3 embed.py dataset=hlsburnscars    encoder=ssl4eo_data2vec preprocessing=seg_resize   task=segmentation
python3 embed.py dataset=hlsburnscars    encoder=ssl4eo_dino preprocessing=seg_resize   task=segmentation
python3 embed.py dataset=hlsburnscars    encoder=ssl4eo_mae_optical preprocessing=seg_resize  task=segmentation
python3 embed.py dataset=hlsburnscars    encoder=ssl4eo_moco preprocessing=seg_resize    task=segmentation

python3 embed.py dataset=hlsburnscars    encoder=unet_encoder preprocessing=seg_resize_input_layer    task=segmentation

python3 embed.py dataset=hlsburnscars    encoder=resnet50_scratch  preprocessing=seg_resize_input_layer    task=segmentation

python3 embed.py dataset=hlsburnscars    encoder=resnet50_pretrained preprocessing=seg_resize_input_layer    task=segmentation

python3 embed.py dataset=hlsburnscars    encoder=vit_scratch preprocessing=seg_resize   task=segmentation

python3 embed.py dataset=hlsburnscars    encoder=vit preprocessing=seg_resize   task=segmentation


