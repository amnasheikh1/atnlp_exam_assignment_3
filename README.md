# Advanced Topics in Natural Language Processing Exam Assignment 3 
## Objective: 
To test the zero-shot generalization capabilities of Pretrained Transformer-based models such as BART trained on SCAN tasks with sequence-to-sequence methods

## Data 

The data comes from a paper by [Lake et al]
(https://doi.org/10.48550/arXiv.1711.00350) and is publicly available. The datasets can be found in the [data](https://github.com/brendenlake/SCAN.git) folder.

The data used for experiment 1 is tasks.txt and for experiment 2 the data used is tasks_train_length and tasks_test_length. 

## Code 

This repository contains all the required code files to run experiment 1 and 2. 
- The files Utils.py, Metrics.py contain helper functions to train and validate codes.
- Dataset.py is used to load and tokenize data for experiment 1.
- Exp_2_Dataset.py is used to load and tokenize the data for experiment 2.
- Transformer.py is custom transformer from Assignment 2 (it is not used in training or validation but it is there to avoid errors while importing files). 
- To train experiment 1: run experiment_1.ipynb. It contains the code to load the data and then train and validate it. 
- To train experiment 2: run experiment_2.ipynb. It contains the code to load the data and then train and validate it.


## Requirements: 
- python==3.10.11
- matplotlib-inline==0.1.7
- pandas==2.2.2
- matplotlib==3.9.0
- seaborn==0.13.2
- pysal==24.1
- datasets==3.2.0
- torch==2.5.1

