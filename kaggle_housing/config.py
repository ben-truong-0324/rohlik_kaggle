import socket

# Determine the hostname
hostname = socket.gethostname()
if hostname == "Khais-MacBook-Pro.local" or hostname == "Khais-MBP.attlocal.net":  
    from kaggle_housing.config_mac import *  
else:
    from kaggle_housing.config_cuda import * 

import os


KAGGLE_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'train.csv')
KAGGLE_TEST_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'test.csv')

DATASET_SELECTION = "kaggle_housing"

EVAL_FUNC_METRIC = 'rmse'  #'f1' # 'accuracy' 

EVAL_MODELS = [
                # 'default',
                'MPL',
                'CNN', 
                'LSTM', 
                'bi-LSTM',
                'conv-LSTM', 
                #'seg-gru',
                ]

PARAM_GRID = {
    'lr': [0.01, 0.005, 0.0005],
    'batch_size': [16, 32],
    
    # 'hidden_layers': [[75,19]],
    'dropout_rate': [0, 0.005, 0.01, ],
    'hidden_layers': [[64, 32], [128, 64, 32], [64],[75]],
    # 'activation_function': just use relu
}


from pathlib import Path
def set_output_dir(path):
    # Ensure the directory exists
    os.makedirs(path, exist_ok=True)
    return path
# Get the root project directory (the parent directory of kaggle_housing)
project_root = Path(__file__).resolve().parent.parent
# Define the output directory path relative to the project root
OUTPUT_DIR_A3 = project_root / 'outputs' / DATASET_SELECTION
DRAFT_VER_A3 = 1
# Set the directories using set_output_dir
AGGREGATED_OUTDIR = set_output_dir(OUTPUT_DIR_A3 / f'ver{DRAFT_VER_A3}_{EVAL_FUNC_METRIC}/aggregated_graphs')
Y_PRED_OUTDIR = set_output_dir(OUTPUT_DIR_A3 / f'ver{DRAFT_VER_A3}_{EVAL_FUNC_METRIC}/y_pred_graphs')
CV_LOSSES_PKL_OUTDIR = set_output_dir(OUTPUT_DIR_A3 / f'ver{DRAFT_VER_A3}_{EVAL_FUNC_METRIC}/pkl_cv')
PERFM_PKL_OUTDIR = set_output_dir(OUTPUT_DIR_A3 / f'ver{DRAFT_VER_A3}_{EVAL_FUNC_METRIC}/perf_pkl')

TXT_OUTDIR = set_output_dir(OUTPUT_DIR_A3 / f'ver{DRAFT_VER_A3}_{EVAL_FUNC_METRIC}/txt_stats')
OUTPUT_DIR_RAW_DATA_A3 =set_output_dir(OUTPUT_DIR_A3 / f'ver{DRAFT_VER_A3}_{EVAL_FUNC_METRIC}/raw_data_assessments')


#ML PARAMS
K_FOLD_CV = 5