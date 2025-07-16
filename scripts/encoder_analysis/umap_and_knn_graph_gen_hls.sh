#!/bin/bash

  
python3 knn_graph_gen.py dataset=hlsburnscars    encoder=terramind_large   decoder=seg_upernet   preprocessing=seg_resize    criterion=cross_entropy    task=segmentation

python3 knn_graph_gen.py dataset=hlsburnscars    encoder=croma_optical    decoder=seg_upernet   preprocessing=seg_resize_input_layer    criterion=cross_entropy    task=segmentation

python3 knn_graph_gen.py dataset=hlsburnscars    encoder=dofa    decoder=seg_upernet   preprocessing=seg_resize_input_layer    criterion=cross_entropy    task=segmentation

python3 knn_graph_gen.py dataset=hlsburnscars    encoder=gfmswin  decoder=seg_upernet   preprocessing=seg_resize  criterion=cross_entropy    task=segmentation

python3 knn_graph_gen.py dataset=hlsburnscars    encoder=prithvi  decoder=seg_upernet   preprocessing=seg_resize_input_layer    criterion=cross_entropy    task=segmentation

python3 knn_graph_gen.py dataset=hlsburnscars    encoder=satlasnet_si  decoder=seg_upernet   preprocessing=seg_resize_input_layer    criterion=cross_entropy    task=segmentation

python3 knn_graph_gen.py dataset=hlsburnscars    encoder=scalemae  decoder=seg_upernet   preprocessing=seg_resize_input_layer    criterion=cross_entropy    task=segmentation

python3 knn_graph_gen.py dataset=hlsburnscars    encoder=spectralgpt decoder=seg_upernet   preprocessing=seg_resize   criterion=cross_entropy    task=segmentation

 
python3 knn_graph_gen.py dataset=hlsburnscars    encoder=ssl4eo_data2vec decoder=seg_upernet   preprocessing=seg_resize_input_layer   criterion=cross_entropy    task=segmentation
python3 knn_graph_gen.py dataset=hlsburnscars    encoder=ssl4eo_dino decoder=seg_upernet   preprocessing=seg_resize_input_layer   criterion=cross_entropy    task=segmentation
python3 knn_graph_gen.py dataset=hlsburnscars    encoder=ssl4eo_mae_optical decoder=seg_upernet   preprocessing=seg_resize_input_layer  criterion=cross_entropy    task=segmentation
python3 knn_graph_gen.py dataset=hlsburnscars    encoder=ssl4eo_moco decoder=seg_upernet   preprocessing=seg_resize_input_layer    criterion=cross_entropy    task=segmentation
 
python3 knn_graph_gen.py dataset=hlsburnscars    encoder=unet_encoder decoder=seg_unet   preprocessing=seg_resize_input_layer    criterion=cross_entropy    task=segmentation

python3 knn_graph_gen.py dataset=hlsburnscars    encoder=resnet50_scratch decoder=seg_unet   preprocessing=seg_resize_input_layer    criterion=cross_entropy    task=segmentation

python3 knn_graph_gen.py dataset=hlsburnscars    encoder=resnet50_pretrained decoder=seg_unet   preprocessing=seg_resize_input_layer    criterion=cross_entropy    task=segmentation


python3 knn_graph_gen.py dataset=hlsburnscars    encoder=vit_scratch decoder=seg_upernet   preprocessing=seg_resize_input_layer   criterion=cross_entropy    task=segmentation

python3 knn_graph_gen.py dataset=hlsburnscars    encoder=vit decoder=seg_upernet   preprocessing=seg_resize_input_layer   criterion=cross_entropy    task=segmentation



