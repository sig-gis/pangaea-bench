#!/bin/bash

python3 weight_watcher.py  dataset=mchesapeake-landcover    encoder=terramind_large   preprocessing=seg_resize    task=segmentation
mkdir ww/teramind/
mv img/* ww/terramind/
mv ww-img/* ww/terramind/
 
python3 weight_watcher.py dataset=mchesapeake-landcover    encoder=croma_optical    preprocessing=seg_resize_input_layer    task=segmentation
mkdir ww/croma/
mv img/* ww/croma/
mv ww-img/* ww/croma/


python3 weight_watcher.py dataset=mchesapeake-landcover    encoder=dofa    preprocessing=seg_resize_input_layer    task=segmentation
mkdir ww/dofa/
mv img/* ww/dofa/
mv ww-img/* ww/dofa/

python3 weight_watcher.py dataset=mchesapeake-landcover    encoder=gfmswin  preprocessing=seg_resize  task=segmentation
mkdir ww/gfmswin/
mv img/* ww/gfmswin/
mv ww-img/* ww/gfmswin/


python3 weight_watcher.py dataset=mchesapeake-landcover    encoder=prithvi  preprocessing=seg_resize_input_layer    task=segmentation
mkdir ww/prithvi/
mv img/* ww/prithvi/
mv ww-img/* ww/prithvi/


python3 weight_watcher.py dataset=mchesapeake-landcover    encoder=satlasnet_si  preprocessing=seg_resize_input_layer    task=segmentation
mkdir ww/satlasnet/
mv img/* ww/satlasnet/
mv ww-img/* ww/satlasnet/


python3 weight_watcher.py dataset=mchesapeake-landcover    encoder=scalemae  preprocessing=seg_resize_input_layer    task=segmentation
mkdir ww/scalemae/
mv img/* ww/scalemae/
mv ww-img/* ww/scalemae/


python3 weight_watcher.py dataset=mchesapeake-landcover    encoder=spectralgpt preprocessing=seg_resize   task=segmentation
mkdir ww/spectralgpt/
mv img/* ww/spectralgpt/
mv ww-img/* ww/spectralgpt/

python3 weight_watcher.py dataset=mchesapeake-landcover    encoder=ssl4eo_data2vec preprocessing=seg_resize   task=segmentation
mkdir ww/ssl4eo_data2vec/
mv img/* ww/ssl4eo_data2vec/
mv ww-img/* ww/ssl4eo_data2vec/
 
python3 weight_watcher.py dataset=mchesapeake-landcover    encoder=ssl4eo_dino preprocessing=seg_resize   task=segmentation
mkdir ww/ssl4eo_dino/
mv img/* ww/ssl4eo_dino/
mv ww-img/* ww/ssl4eo_dino/

python3 weight_watcher.py dataset=mchesapeake-landcover    encoder=ssl4eo_mae_optical preprocessing=seg_resize  task=segmentation
mkdir ww/ssl4eo_mae/
mv img/* ww/ssl4eo_mae/
mv ww-img/* ww/ssl4eo_mae/

python3 weight_watcher.py dataset=mchesapeake-landcover    encoder=ssl4eo_moco preprocessing=seg_resize    task=segmentation
mkdir ww/ssl4eo_moco/
mv img/* ww/ssl4eo_moco/
mv ww-img/* ww/ssl4eo_moco/ 

python3 weight_watcher.py dataset=mchesapeake-landcover    encoder=unet_encoder preprocessing=seg_resize_input_layer    task=segmentation
mkdir ww/unet/
mv img/* ww/unet/
mv ww-img/* ww/unet/
 
python3 weight_watcher.py dataset=mchesapeake-landcover    encoder=resnet50_scratch preprocessing=seg_resize_input_layer    task=segmentation
mkdir ww/resnet50_scratch/
mv img/* ww/resnet50_scratch/
mv ww-img/* ww/resnet50_scratch/

python3 weight_watcher.py dataset=mchesapeake-landcover    encoder=resnet50_pretrained preprocessing=seg_resize_input_layer    task=segmentation
mkdir ww/resnet50/
mv img/* ww/resnet50/
mv ww-img/* ww/resnet50/

python3 weight_watcher.py dataset=mchesapeake-landcover    encoder=vit_scratch preprocessing=seg_resize   task=segmentation
mkdir ww/vit_scratch/
mv img/* ww/vit_scratch/
mv ww-img/* ww/vit_scratch/

python3 weight_watcher.py dataset=mchesapeake-landcover    encoder=vit preprocessing=seg_resize   task=segmentation
mkdir ww/vit/
mv img/* ww/vit/
mv ww-img/* ww/vit/ 




