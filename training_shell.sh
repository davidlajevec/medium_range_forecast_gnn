#!/bin/bash

# Define constants
TRAINING_NAME="gcn3"
BATCH_SIZE=32
EPOCHS=15
VARIABLES="geopotential_500,u_500,v_500"
HIDDEN_CHANNELS=16
LR=0.01
GAMMA=0.7
START_YEAR_TRAINING=1950
END_YEAR_TRAINING=2002
START_YEAR_VALIDATION = 2003
END_YEAR_VALIDATION = 2014
START_YEAR_TEST = 2021
END_YEAR_TEST = 2021

# Run main.py with the defined constants
python main.py --training_name $TRAINING_NAME \
               --batch_size $BATCH_SIZE \
               --epochs $EPOCHS \
               --variables $VARIABLES \
               --num_variables $NUM_VARIABLES \
               --hidden_channels $HIDDEN_CHANNELS \
               --lr $LR \
               --gamma $GAMMA \
               --start_year $START_YEAR \
               --end_year $END_YEAR \
               --start_year_validation $START_YEAR_VALIDATION \
               --end_year_validation $END_YEAR_VALIDATION