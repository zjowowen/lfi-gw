#!/bin/bash

python -m lfigw.gwpe train new ffjord \
    --data_dir waveforms/GW150914/ \
    --model_dir models/GW150914/ \
    --lr 0.0002 \
    --epochs 1 \
    --distance_prior_fn uniform_distance \
    --distance_prior 100.0 1000.0 \
    --truncate_basis 100 \
    --activation elu \
    --lr_anneal_method cosine \
    --batch_size 9000
