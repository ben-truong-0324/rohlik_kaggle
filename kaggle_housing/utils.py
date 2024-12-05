
import re
import time
import os
from datetime import datetime
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate 
import pickle
import random
from copy import deepcopy
import kaggle_housing.hypotheses
from src.models import *
from src.models_reg import *
import math

from scipy.stats import ttest_1samp
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix,precision_score, \
                        recall_score,classification_report, \
                        accuracy_score, f1_score, log_loss, \
                       confusion_matrix, ConfusionMatrixDisplay,\
                          roc_auc_score, matthews_corrcoef, average_precision_score
from sklearn.cluster import KMeans, AgglomerativeClustering,DBSCAN,Birch,MeanShift, SpectralClustering

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import ParameterSampler

#import dimension reduction modules
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold


from torch import nn, optim
import torch
from torch.utils.data import DataLoader, TensorDataset


def set_random_seed(seed): #use for torch nn training in MC simulation
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # Set seed for all GPUs
        torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
        torch.backends.cudnn.benchmark = False  

def set_output_dir(outpath):
    os.makedirs(outpath, exist_ok=True)
    return outpath

def purity_score(y_true, y_pred):
    # Matrix of contingency
    contingency_matrix = np.zeros((len(set(y_true)), len(set(y_pred))))
    for i, label in enumerate(y_true):
        contingency_matrix[label, y_pred[i]] += 1
    # Take the max label count for each cluster, sum them, and divide by total samples
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def get_metrics_of_hyperparm_set(y_preds):
    accuracies = []
    mcc_scores = []
    f1_scores = []
    roc_auc_scores = []
    pr_auc_scores = []

    # Iterate over all y_preds, assuming y_preds is a list of tuples (y_test, predicted)
    for y_test, predicted in y_preds:
        # Compute matches and row accuracies
        matches = (y_test == predicted)
        row_accuracies = np.mean(matches, axis=1)  # Mean across each row (for multi-label)
        
        # Compute overall accuracy for the current fold (mean of row accuracies)
        accuracies.append(np.mean(row_accuracies))

        # Compute MCC (Matthews Correlation Coefficient) for the current fold
        mcc = matthews_corrcoef(y_test.flatten(), predicted.flatten())  # Flatten if multi-label
        mcc_scores.append(mcc)

        # Compute F1 Score (for binary or multi-label)
        f1 = f1_score(y_test, predicted, average='macro')  # 'macro' averages F1 score across labels
        f1_scores.append(f1)

        # Compute AUC-ROC (Area Under the Receiver Operating Characteristic Curve)
        auc_roc = roc_auc_score(y_test, predicted, average='macro', multi_class='ovr')  # 'ovr' for one-vs-rest
        roc_auc_scores.append(auc_roc)

        # Compute AUC-PR (Area Under the Precision-Recall Curve)
        auc_pr = average_precision_score(y_test, predicted, average='macro')  # 'macro' averages PR score across labels
        pr_auc_scores.append(auc_pr)

    # Calculate the average of all metrics across folds
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)  # Standard deviation for accuracy

    avg_mcc = np.mean(mcc_scores)
    avg_f1 = np.mean(f1_scores)
    avg_roc_auc = np.mean(roc_auc_scores)
    avg_pr_auc = np.mean(pr_auc_scores)
    return avg_accuracy, std_accuracy, avg_mcc, avg_f1, avg_roc_auc, avg_pr_auc



# Evaluation functions
def calculate_fin_metrics(y_pred, y_actual):
    num_samples, num_labels = y_pred.shape
    
    # Initialize lists to store individual metrics for each label
    label_accuracy = []
    auc_roc = []
    f1 = []
    mcc = []
    auprc = []
    
    for i in range(num_labels):
        # Calculate accuracy for each label
        acc = (y_pred[:, i] == y_actual[:, i]).mean()  # Mean accuracy for each label
        label_accuracy.append(acc)
        
        # Handle the AUC-ROC calculation only if both classes are present
        try:
            auc = roc_auc_score(y_actual[:, i], y_pred[:, i])
            auc_roc.append(auc)
        except ValueError:
            # Skip if only one class is present in y_true
            auc_roc.append(np.nan)
        
        # Calculate F1 score for each label
        f1_score_label = f1_score(y_actual[:, i], y_pred[:, i])
        f1.append(f1_score_label)
        
        # Calculate Matthews Correlation Coefficient (MCC) for each label
        mcc_score = matthews_corrcoef(y_actual[:, i], y_pred[:, i])
        mcc.append(mcc_score)
        
        # Calculate Average Precision (AU-PRC) for each label
        auprc_score = average_precision_score(y_actual[:, i], y_pred[:, i])
        auprc.append(auprc_score)
    
    # Overall accuracy for all 19 labels (mean of the per-label accuracies)
    overall_accuracy = np.mean(label_accuracy)
    
    return label_accuracy, auc_roc, f1, mcc, auprc, overall_accuracy

import pandas as pd
import numpy as np

def print_and_save_metrics(results, output_file):
    """
    Save and print the mean and standard deviation for each metric (excluding 'overall_accuracy')
    for each dataset and model.

    :param results: A dictionary containing dataset metrics with model names and their respective metric values.
    :param output_file: Path to the file where metrics will be saved.
    """
    print("hello")
    with open(output_file, 'w') as f:
        for dataset, models_metrics in results.items():
            f.write(f"Dataset: {dataset}\n")
            
            # Extracting the metrics for each model
            metrics_to_process = ['label_accuracy', 'auc_roc', 'f1', 'mcc', 'auprc']
            dataset_stats = {}
            model_stats = {}

            # Compute mean and std for each metric, per model
            for model_metrics in models_metrics:
                model_name = model_metrics["model"]
                model_stats[model_name] = {}

                for metric in metrics_to_process:
                    metric_values = model_metrics[metric]
                    mean_value = np.mean(metric_values)
                    std_value = np.std(metric_values)
                    model_stats[model_name][metric] = {"mean": mean_value, "std": std_value}
            
            # Now prepare the output for dataset-level statistics
            dataset_level_stats = {}

            for metric in metrics_to_process:
                # Gather all model mean values for the metric to calculate dataset-level mean and std
                model_mean_values = [model_stats[model][metric]["mean"] for model in model_stats]
                model_std_values = [model_stats[model][metric]["std"] for model in model_stats]

                dataset_level_stats[metric] = {
                    "mean": np.mean(model_mean_values),
                    "std": np.std(model_mean_values)
                }

            # Write out dataset-level stats
            f.write("Dataset-level statistics (mean and std of each model's metrics):\n")
            dataset_df = pd.DataFrame(dataset_level_stats).T
            dataset_df.index.name = 'Metric'
            f.write(dataset_df.to_string())
            f.write("\n\n")

            # Write out model-level stats (mean and std for each model and metric)
            f.write("Model-level statistics (mean and std for each model):\n")
            for model_name, stats in model_stats.items():
                f.write(f"Model: {model_name}\n")
                model_df = pd.DataFrame(stats).T
                model_df.index.name = 'Metric'
                f.write(model_df.to_string())
                f.write("\n\n")

            # Print the statistics to the console
            print(f"Metrics for dataset {dataset}:")
            print(dataset_df)
            for model_name, stats in model_stats.items():
                print(f"Model: {model_name}")
                print(pd.DataFrame(stats).T)

# Example usage
# print_and_save_metrics(results, 'metrics_summary.txt')


def check_data_info(X, y, X_train, X_test, y_train, y_test, show = False):
    if show:
        # Check data types and shapes for each of the variables
        data_info = {
            'X': {'dtype': type(X), 'shape': X.shape if isinstance(X, (np.ndarray, pd.DataFrame)) else 'Not an array-like object'},
            'y': {'dtype': type(y), 'shape': y.shape if isinstance(y, (np.ndarray, pd.Series)) else 'Not an array-like object'},
            'X_train': {'dtype': type(X_train), 'shape': X_train.shape if isinstance(X_train, (np.ndarray, pd.DataFrame)) else 'Not an array-like object'},
            'X_test': {'dtype': type(X_test), 'shape': X_test.shape if isinstance(X_test, (np.ndarray, pd.DataFrame)) else 'Not an array-like object'},
            'y_train': {'dtype': type(y_train), 'shape': y_train.shape if isinstance(y_train, (np.ndarray, pd.Series)) else 'Not an array-like object'},
            'y_test': {'dtype': type(y_test), 'shape': y_test.shape if isinstance(y_test, (np.ndarray, pd.Series)) else 'Not an array-like object'}
        }

        # Print the data type and shape for each variable
        for var, info in data_info.items():
            print(f'{var}: Type = {info["dtype"]}, Shape = {info["shape"]}')
            
        # Function to check columns, data types, and unique values
        def check_dataframe_info(df, name):
            if isinstance(df, (pd.DataFrame, pd.Series)):
                print(f"\n{name} DataFrame/Series:")
                # Iterate through columns to check unique counts
                if isinstance(df, pd.DataFrame):  # For DataFrame, check columns
                    for col in df.columns:
                        if df[col].dtype in ['int64', 'float64']:  # Numeric column
                            if df[col].nunique() > 10:  # Skip unique count if there are more than 10 unique values
                                print(f"Column '{col}': {df[col].dtype}, Numerical")
                            else:
                                print(f"Column '{col}': {df[col].dtype}, Unique Values = {df[col].nunique()}")
                        else:  # Categorical column (non-numeric)
                            print(f"Column '{col}': {df[col].dtype}, Unique Values = {df[col].nunique()}")
                else:  # For Series, just show unique counts
                    print(f"Unique Values: {df.nunique()}")

        # Check DataFrame info for X, X_train, and X_test (assuming they are DataFrames)
        if isinstance(X, (pd.DataFrame, pd.Series)):
            check_dataframe_info(X, 'X')
        if isinstance(X_train, (pd.DataFrame, pd.Series)):
            check_dataframe_info(X_train, 'X_train')
        if isinstance(X_test, (pd.DataFrame, pd.Series)):
            check_dataframe_info(X_test, 'X_test')
        if isinstance(y, (pd.Series, pd.DataFrame)):
            check_dataframe_info(y, 'y')
        if isinstance(y_train, (pd.Series, pd.DataFrame)):
            check_dataframe_info(y_train, 'y_train')
        if isinstance(y_test, (pd.Series, pd.DataFrame)):
            check_dataframe_info(y_test, 'y_test')





def train_nn_early_stop_regression(X_train, y_train, X_test, y_test, device,criterion, max_epochs, patience, model_name="default"):
    input_dim = X_train.shape[1]

    if isinstance(criterion, (nn.MSELoss, nn.L1Loss)):  # Check if the criterion is a regression loss
        output_dim = 1  # Regression tasks always have a single continuous output
    else:
        if len(y_train.shape) > 1 and y_train.shape[1] > 1:
            # Multi-label classification (y_train has multiple labels per instance)
            output_dim = y_train.shape[1]  # Number of labels
        else:
            # Single-label classification (y_train has a single label per instance)
            output_dim = len(np.unique(y_train.cpu()))
    

    if model_name == "MPL":
        model = MLPRegression(input_dim, output_dim,).to(device)
    elif model_name == "MPL_lessrelu":
        model = MLPRegression_lessRelu(input_dim, output_dim,).to(device)
    
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.to(device)
    criterion.to(device)

    # Initialize early stopping variables
    best_loss = float("inf")
    patience_counter = 0
    epoch_losses = []

    # Training loop
    start_time = time.time()
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass for training
        outputs_train = model(X_train)
        train_loss = criterion(outputs_train, y_train)
        
        # Backward pass
        train_loss.backward()
        optimizer.step()

        

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            outputs_eval = model(X_test)
            eval_loss = criterion(outputs_eval, y_test)

        # Early stopping logic
        if eval_loss.item() < best_loss:
            best_loss = eval_loss.item()
            patience_counter = 0
            best_model_state = model.state_dict()  # Save the best model
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

        # Store both training and evaluation losses
        epoch_losses.append({
            "epoch": epoch + 1,
            "train_loss": train_loss.item(),
            "eval_loss": eval_loss.item()
        })

    # Restore the best model
    model.load_state_dict(best_model_state)

    runtime = time.time() - start_time
    epoch_trained = epoch + 1

    print(model)
    print(f"Model {model_name} Training completed in {runtime // 60:.0f}m {runtime % 60:.0f}s")
    print(f"Average time per epoch: {(runtime / epoch_trained) // 60:.0f}m {(runtime / epoch_trained) % 60:.0f}s")

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        outputs = outputs.cpu().numpy()
        y_test = y_test.cpu().numpy()

    # Compute regression metrics
    mse = mean_squared_error(y_test, outputs)
    mae = mean_absolute_error(y_test, outputs)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, outputs)

    print(f"Training terminated after epoch {epoch_trained}, "
          f"MSE: {mse:.4f}, "
          f"MAE: {mae:.4f}, "
          f"RMSE: {rmse:.4f}, "
          f"R²: {r2:.4f}")

    # Return metrics and model
    return mse, mae, rmse, r2, runtime, model, outputs, epoch_losses