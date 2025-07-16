#!/bin/bash

#python3 weight_watcher.py  dataset=hlsburnscars    encoder=terramind_large   decoder=seg_upernet   preprocessing=seg_resize    criterion=cross_entropy    task=segmentation
 
#python3 weight_watcher.py dataset=hlsburnscars    encoder=croma_optical    decoder=seg_upernet   preprocessing=seg_resize_input_layer    criterion=cross_entropy    task=segmentation
#mkdir ww/croma/
#mv img/* ww/croma/
#mv ww-img/* ww/croma/


#python3 weight_watcher.py dataset=hlsburnscars    encoder=dofa    decoder=seg_upernet   preprocessing=seg_resize_input_layer    criterion=cross_entropy    task=segmentation
#mkdir ww/dofa/
#mv img/* ww/dofa/
#mv ww-img/* ww/dofa/

#python3 weight_watcher.py dataset=hlsburnscars    encoder=gfmswin  decoder=seg_upernet   preprocessing=seg_resize  criterion=cross_entropy    task=segmentation
#mkdir ww/gfmswin/
#mv img/* ww/gfmswin/
#mv ww-img/* ww/gfmswin/


#python3 weight_watcher.py dataset=hlsburnscars    encoder=prithvi  decoder=seg_upernet   preprocessing=seg_resize_input_layer    criterion=cross_entropy    task=segmentation
#mkdir ww/prithvi/
#mv img/* ww/prithvi/
#mv ww-img/* ww/prithvi/


#python3 weight_watcher.py dataset=hlsburnscars    encoder=satlasnet_si  decoder=seg_upernet   preprocessing=seg_resize_input_layer    criterion=cross_entropy    task=segmentation
#mkdir ww/satlasnet/
#mv img/* ww/satlasnet/
#mv ww-img/* ww/satlasnet/


#python3 weight_watcher.py dataset=hlsburnscars    encoder=scalemae  decoder=seg_upernet   preprocessing=seg_resize_input_layer    criterion=cross_entropy    task=segmentation
#mkdir ww/scalemae/
#mv img/* ww/scalemae/
#mv ww-img/* ww/scalemae/


#python3 weight_watcher.py dataset=hlsburnscars    encoder=spectralgpt decoder=seg_upernet   preprocessing=seg_resize   criterion=cross_entropy    task=segmentation
#mkdir ww/spectralgpt/
#mv img/* ww/spectralgpt/
#mv ww-img/* ww/spectralgpt/

#python3 weight_watcher.py dataset=hlsburnscars    encoder=ssl4eo_data2vec decoder=seg_upernet   preprocessing=seg_resize   criterion=cross_entropy    task=segmentation
#mkdir ww/ssl4eo_data2vec/
#mv img/* ww/ssl4eo_data2vec/
#mv ww-img/* ww/ssl4eo_data2vec/
 
#python3 weight_watcher.py dataset=hlsburnscars    encoder=ssl4eo_dino decoder=seg_upernet   preprocessing=seg_resize   criterion=cross_entropy    task=segmentation
#mkdir ww/ssl4eo_dino/
#mv img/* ww/ssl4eo_dino/
#mv ww-img/* ww/ssl4eo_dino/

#python3 weight_watcher.py dataset=hlsburnscars    encoder=ssl4eo_mae_optical decoder=seg_upernet   preprocessing=seg_resize  criterion=cross_entropy    task=segmentation
#mkdir ww/ssl4eo_mae/
#mv img/* ww/ssl4eo_mae/
#mv ww-img/* ww/ssl4eo_mae/

#python3 weight_watcher.py dataset=hlsburnscars    encoder=ssl4eo_moco decoder=seg_upernet   preprocessing=seg_resize    criterion=cross_entropy    task=segmentation
#mkdir ww/ssl4eo_moco/
#mv img/* ww/ssl4eo_moco/
#mv ww-img/* ww/ssl4eo_moco/ 

#python3 weight_watcher.py dataset=hlsburnscars    encoder=unet_encoder decoder=seg_unet   preprocessing=seg_resize_input_layer    criterion=cross_entropy    task=segmentation
#mkdir ww/unet/
#mv img/* ww/unet/
#mv ww-img/* ww/unet/
 
python3 weight_watcher.py dataset=hlsburnscars    encoder=resnet50_scratch decoder=seg_upernet   preprocessing=seg_resize_input_layer    criterion=cross_entropy    task=segmentation
mkdir ww/resnet50_scratch/
mv img/* ww/resnet50_scratch/
mv ww-img/* ww/resnet50_scratch/

python3 weight_watcher.py dataset=hlsburnscars    encoder=resnet50_pretrained decoder=seg_upernet   preprocessing=seg_resize_input_layer    criterion=cross_entropy    task=segmentation
mkdir ww/resnet50/
mv img/* ww/resnet50/
mv ww-img/* ww/resnet50/

#python3 weight_watcher.py dataset=hlsburnscars    encoder=vit_scratch decoder=seg_upernet   preprocessing=seg_resize   criterion=cross_entropy    task=segmentation
#mkdir ww/vit_scratch/
#mv img/* ww/vit_scratch/
#mv ww-img/* ww/vit_scratch/

#python3 weight_watcher.py dataset=hlsburnscars    encoder=vit decoder=seg_upernet   preprocessing=seg_resize   criterion=cross_entropy    task=segmentation
#mkdir ww/vit/
#mv img/* ww/vit/
#mv ww-img/* ww/vit/ 




