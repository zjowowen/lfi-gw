#!/bin/bash

# This trains a neural conditional density estimator with a neural spline
# coupling flow.

# Settings are the same as in the paper. It will take several days to run, so
# you may want to decrease the number of epochs or the size of the network.

# Feel free to change the settings, but only the nde flow option will work at
# present.

python -m train_ffjord train new ffjord \
    --data_dir waveforms/GW150914/ \
    --model_dir models/GW150914/ \
    --lr 0.0002 \
    --epochs 500 \
    --distance_prior_fn uniform_distance \
    --distance_prior 100.0 1000.0 \
    --truncate_basis 100 \
    --activation elu \
    --lr_anneal_method cosine \
    --batch_size 4096
