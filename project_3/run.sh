#! /bin/bash

############################################################################################
# 4 arguments:                                                                             #
# 1. --demo: to skip training                                                             #
# 2. --model: to specify which model to train or load; default: 5_Layer                    #
# 3. --train_data: to specify which data set to use to train themodel; default: pokemon    #
# 4. --test_data: to specify which dataset to use to test during demo; default:cats         #
############################################################################################

echo "running python code..."
python src/create_noisy.py
python src/train.py --model 1_Layer
python src/train.py --model 1_Layer --demo
python src/train.py --model 5_Layer
python src/train.py --model 5_Layer --demo
python src/train.py --model 5_Layer_Relu
python src/train.py --model 5_Layer_Relu --demo


