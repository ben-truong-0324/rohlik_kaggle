from kaggle_housing.config import *
from src.models import *
from kaggle_housing.utils import *
from kaggle_housing.tests import *

import kaggle_housing.etl as etl
import kaggle_housing.plots as plots
import kaggle_housing.hypotheses as hypotheses

import pickle
import glob
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
import math
import itertools

from scipy.stats import ttest_1samp

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

import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def train_nn_with_early_stopping_with_param(X_train, y_train, X_test, y_test, params, max_epochs, patience, model_name="default"):
    lr = params['lr']
    batch_size = params['batch_size']
    hidden_layers = params['hidden_layers']
    dropout_rate = params['dropout_rate']
    input_dim = X_train.shape[1]
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        # Multi-label classification (y_train has multiple labels per instance)
        output_dim = y_train.shape[1]  # Number of labels
    else:
        # Single-label classification (y_train has a single label per instance)
        output_dim = len(np.unique(y_train.cpu())) 
    if model_name == "default":
        model = SimpleNN(input_dim, output_dim, hidden_layers, dropout_rate=dropout_rate).to(device)
    elif model_name == "MPL":
        model = FarsightMPL(input_dim=input_dim, output_dim=output_dim).to(device)
    elif model_name == "CNN":
        model = FarsightCNN(input_dim=input_dim, output_dim=output_dim,hidden_dim=289, feature_maps=19, dropout_rate=params['dropout_rate']).to(device)

    elif model_name == "LSTM":
        model = FarsightLSTM(input_dim=input_dim, output_dim=output_dim,hidden_dim=289, lstm_hidden_dim=300, dropout_rate=params['dropout_rate']).to(device)

    elif model_name == "bi-LSTM":
        model = FarsightBiLSTM(input_dim=input_dim, output_dim=output_dim,  hidden_dim=289, lstm_hidden_dim=150, dropout_rate=params['dropout_rate']).to(device)

    elif model_name == "conv-LSTM":
        model = FarsightConvLSTM(input_dim=input_dim, output_dim=output_dim,  hidden_dim=289, feature_maps=19, lstm_hidden_dim=300, dropout_rate=params['dropout_rate']).to(device)

    else:
        raise ValueError(f"Unsupported model type: {model_name}")
    # model = FarsightMPL(input_dim, output_dim, dropout_rate).to(device)



    if len(y_train.shape) == 1:  
        criterion = nn.CrossEntropyLoss()  # For single label classification (multi-class)
    else:  
        criterion = nn.BCEWithLogitsLoss()  # For multi-label classification

    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    epoch_trained = 0

    epoch_losses = []
    start_time = time.time()
    print("Starting training loop...")
    for epoch in range(max_epochs):
        epoch_start_time = time.time()
        epoch_trained+=1
        model.train()

        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_epoch_loss = running_loss / len(train_loader)
        # Validation
        val_loss = evaluate_model(model, X_test, y_test, device,criterion)
        epoch_losses.append((avg_epoch_loss,val_loss))
        print(f"Epoch {epoch}, last train_loss {epoch_losses[-1][0]:.5F} val_loss {val_loss:.5F} per {criterion}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break
        epoch_runtime = time.time() - epoch_start_time
        print(f"Epoch completed in {epoch_runtime // 60:.0f}m {epoch_runtime % 60:.0f}s\n")
    runtime = time.time() - start_time
    print(model)
    print(f"Model {model_name} Training completed in {runtime // 60:.0f}m {runtime % 60:.0f}s\n")
    print(f"Average time per epoch: {(runtime / epoch_trained )//60:.0f}m {(runtime / epoch_trained)%60:.0f}s")

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)

        _, predicted = torch.max(outputs, 1)

        if y_train.shape[1] > 1:
            test_outputs_np = outputs.sigmoid().cpu().numpy()  # Sigmoid for multi-label probability
           
            y_test = y_test.cpu().numpy()
            best_threshold = .5
            predicted = (test_outputs_np >= best_threshold).astype(int)
            # Calculate accuracy, AUC-ROC, and F1-score for multi-label classification
            # Individual label accuracy (mean accuracy for each label)
            label_accuracy = (predicted == y_test).mean(axis=0)
            accuracy = label_accuracy.mean()  # Average accuracy across all labels
            try:
                auc_roc = roc_auc_score(y_test, test_outputs_np, average="macro")
            except ValueError:
                auc_roc = float("nan")  # Handle cases where AUC-ROC can't be calculated
            f1 = f1_score(y_test, predicted, average="macro")
            mcc = matthews_corrcoef(y_test.flatten(), predicted.flatten())
            auprc = average_precision_score(y_test, test_outputs_np, average="macro")

        #if y is single label
        else:
            accuracy = accuracy_score(y_test.cpu(), predicted.cpu())
            f1 = f1_score(y_test.cpu(), predicted.cpu(), average='weighted')
            # probs = torch.softmax(outputs, dim=1)
            # auc_roc = roc_auc_score(y_test.cpu(), probs.cpu(), multi_class='ovr')  # For multi-class problems
            probs = torch.sigmoid(outputs)[:, 0]  # Assuming the positive class is the first one
            auc_roc = roc_auc_score(y_test.cpu(), probs.cpu())
            mcc = matthews_corrcoef(y_test.cpu(), predicted.cpu())
            auprc = average_precision_score(y_test.cpu(), probs.cpu())


        ##################
    print(f"Training terminated after epoch {epoch_trained}, "
            f"Avg Label Accuracy: {accuracy:.4f}, "
            f"AUC-ROC: {auc_roc:.4f}, "
            f"F1-Score: {f1:.4f}, "
            f"MCC: {mcc:.4f}, "
            f"AU-PRC: {auprc:.4f}")


    
    return accuracy, f1,auc_roc, mcc, auprc, runtime,model,epoch_losses,y_test,predicted


def get_eval_reg_with_nn(X,y,nn_performance_path,cv_losses_outpath, y_pred_outpath, do_cv = 1):
    if not os.path.exists(nn_performance_path) or not os.path.exists(cv_losses_outpath) or not os.path.exists(y_pred_outpath):
        X = pd.DataFrame(X)  # Assuming X_train is a DataFrame
        X = torch.FloatTensor(X.values).to(device)
        y = torch.FloatTensor(y).to(device)
        nn_results={}

        ###################################
        for model_name in EVAL_REG_MODELS:
            

            # avg_metric_per_cv = [0 for _ in range(K_FOLD_CV)] if do_cv else [0]
            # Initialize placeholders for cross-validation metrics
            avg_metrics_per_cv = {
                "mse": [],
                "mae": [],
                "rmse": [],
                "r2": [],
                "runtime": []
            }
            cv_losses = []
            y_preds = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X) if do_cv else [(range(len(X)), range(len(X)))]):
                print(f"Starting fold {fold_idx + 1}")
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                best_model, best_params,best_r2,log_rmse = reg_hyperparameter_tuning(X_train, y_train, X_val, y_val, device, model_name)

                
               

    #             # Log epoch losses and predictions
    #             cv_losses.append(epoch_losses)
    #             y_preds.append({"y_true": y_val.tolist(), "y_pred": outputs.tolist()})
    #         # Calculate average metrics across folds
    #         nn_results[model_name] = {
    #             "avg_mse": np.mean(avg_metrics_per_cv["mse"]),
    #             "avg_mae": np.mean(avg_metrics_per_cv["mae"]),
    #             "avg_rmse": np.mean(avg_metrics_per_cv["rmse"]),
    #             "avg_r2": np.mean(avg_metrics_per_cv["r2"]),
    #             "avg_runtime": np.mean(avg_metrics_per_cv["runtime"]),
    #             "cv_losses": cv_losses,
    #             "y_preds": y_preds,
    #         }

    #         # Print summary for the current model
    #         print(f"Model {model_name} results:")
    #         print(f"  Average MSE: {nn_results[model_name]['avg_mse']:.4f}")
    #         print(f"  Average MAE: {nn_results[model_name]['avg_mae']:.4f}")
    #         print(f"  Average RMSE: {nn_results[model_name]['avg_rmse']:.4f}")
    #         print(f"  Average R²: {nn_results[model_name]['avg_r2']:.4f}")
    #         print(f"  Average Runtime: {nn_results[model_name]['avg_runtime']:.2f}s")

    #     # with open(nn_performance_path, "wb") as f:
    #     #     pickle.dump(nn_results, f, )
    #     # with open(cv_losses_outpath, "wb") as f:
    #     #     pickle.dump(cv_losses, f, )
    #     # with open(y_pred_outpath, "wb") as f:
    #     #     pickle.dump(y_preds, f,)
    # else:
    #     with open(nn_performance_path, 'rb') as f:
    #         nn_results = pickle.load(f)
    #     with open(cv_losses_outpath, 'rb') as f:
    #         cv_losses = pickle.load(f)
    #     with open(y_pred_outpath, 'rb') as f:
    #         y_preds = pickle.load(f)
    # return nn_results, cv_losses, y_preds



        


def get_eval_with_nn(X,y,nn_performance_path,cv_losses_outpath, y_pred_outpath, do_cv = 1):
    if not os.path.exists(nn_performance_path) or not os.path.exists(cv_losses_outpath):
        X = pd.DataFrame(X)  # Assuming X_train is a DataFrame
        X = torch.FloatTensor(X.values).to(device)
        y = torch.FloatTensor(y).to(device)
        nn_results={}

        param_combinations = list(itertools.product(*PARAM_GRID.values()))

        ###################################
        for model_name in EVAL_MODELS:
            for params in param_combinations:
                current_params = {
                    'lr': params[0],
                    'batch_size': params[1],
                    'dropout_rate': params[2],
                    'hidden_layers': params[3],
                }

                current_metrics_of_Xy = []

                avg_metric_per_cv = [0 for _ in range(K_FOLD_CV)] if do_cv else [0]
                cv_losses = []
                y_preds = []
                
                for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X) if do_cv else [(range(len(X)), range(len(X)))]):
                    print(f"Starting fold {fold_idx + 1}")
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    # Train and evaluate model with current parameters
                    accuracy, f1,auc_roc,mcc, auprc, runtime,temp_model,epoch_losses,y_test,predicted = train_nn_with_early_stopping_with_param(
                        X_train, y_train, X_val, y_val, current_params,NN_MAX_EPOCH, NN_PATIENCE, model_name,
                    )
                    
                    # Store the current metrics
                    current_metrics_of_Xy.append((accuracy, f1, runtime, auc_roc,mcc, auprc))
                    
                    # Choose evaluation metric
                    if "f1" in EVAL_FUNC_METRIC:
                        avg_metric_per_cv[fold_idx] = f1
                    elif "accuracy" in EVAL_FUNC_METRIC:
                        avg_metric_per_cv[fold_idx] = accuracy
                    elif "auc" in EVAL_FUNC_METRIC:
                        avg_metric_per_cv[fold_idx] = auc_roc
                    cv_losses.append(epoch_losses)
                    y_preds.append((y_test,predicted))

            # Calculate average metric across folds
            avg_metric = np.mean(avg_metric_per_cv)
            
            # Update running best if the new metric is better
            if avg_metric > inner_cv_running_best_metric:
                inner_cv_running_best_metric = avg_metric

            if inner_cv_running_best_metric > outer_ro_running_best_metric:
                outer_ro_running_best_metric = inner_cv_running_best_metric
                running_best_y_preds = y_preds
                avg_accuracy, std_accuracy, avg_mcc, avg_f1, avg_roc_auc, avg_pr_auc = get_metrics_of_hyperparm_set(y_preds)
            
                running_best_result_dict = {
                    'model_name': model_name,
                    'avg_accuracy': avg_accuracy,
                    'std_accuracy': std_accuracy,  # Save the standard deviation for accuracy
                    'avg_mcc': avg_mcc,
                    'avg_f1': avg_f1,
                    'avg_roc_auc': avg_roc_auc,
                    'avg_pr_auc': avg_pr_auc,
                    'max_epoch': NN_MAX_EPOCH,
                    'current_params': current_params,
                    'current_metrics_of_Xy': current_metrics_of_Xy,
                    'y_preds': y_preds,
                    'cv_losses': cv_losses,
                    
                }
            
                with open(stats_filename, 'w') as f:
                    f.write(f"Model: {model_name}\n")
                    f.write(f"Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}\n")
                    f.write(f"Average MCC: {avg_mcc:.4f}\n")
                    f.write(f"Average F1 Score: {avg_f1:.4f}\n")
                    f.write(f"Average AUC-ROC: {avg_roc_auc:.4f}\n")
                    f.write(f"Average AUC-PR: {avg_pr_auc:.4f}\n")
                    f.write(f"max_epoch: {NN_MAX_EPOCH}\n")
                    f.write(f"Hyperparameters: {current_params}\n")
                print(f"Saved stats to {stats_filename}")

            best_overall_metric, best_overall_model, best_overall_method, running_metrics_Xy_srx_space, \
                best_overall_cv_losses,running_best_y_preds,running_best_result_dict = run_model_tuning_RO_for_Xy_srx_space(
                    X_features, 
                    y_labels, 
                    do_cv=True, 
                    random_opt_algo="default", 
                    best_overall_metric=best_overall_metric,  # Keyword argument
                    best_overall_method=best_overall_method,    # Keyword argument
                    best_overall_model=best_overall_model,    # Keyword argument
                    best_overall_cv_losses = best_overall_cv_losses,
                    type_tag=f"farsight_{model_name}",             # Keyword argument,
                    model_name = model_name,
                )
            nn_results[model_name] = {'mc_results': running_metrics_Xy_srx_space}
            with open(f'{NN_PKL_OUTDIR}/farsight_best_{model_name}_hyperparam_set.pkl', 'wb') as f:
                pickle.dump(running_best_result_dict,f)
            with open(f'{Y_PRED_PKL_OUTDIR}/y_pred_best_of_{model_name}.pkl', 'wb') as f:
                pickle.dump(running_best_y_preds,f)
            print(f"Saved results to {Y_PRED_PKL_OUTDIR}/y_pred_best_of_{model_name}.pkl")
        with open(f'{NN_PKL_OUTDIR}/farsight_best_of_{model_name}_nn_results.pkl', 'wb') as f:
            pickle.dump(nn_results,f)
        print(f"Saved results to {NN_PKL_OUTDIR}/farsight_best_of_{model_name}_nn_results.pkl")

def evaluate_metrics_in_context(y_true, y_pred, model_name, file_path=f"{TXT_OUTDIR}/dt_model_results.txt"):
    # Calculate MSE, MAE, and R²
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate average price for relative error calculations
    avg_price = np.mean(y_true)
    
    # Calculate relative MSE and MAE
    relative_mse = (mse / avg_price) * 100
    relative_mae = (mae / avg_price) * 100
    
    # Print results in context
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R² Score: {r2}")
    print(f"Relative MSE (% of avg price): {relative_mse:.2f}%")
    print(f"Relative MAE (% of avg price): {relative_mae:.2f}%")
    
    # Append the results to the file
    with open(file_path, "a") as log_file:
        log_file.write(f"\nFor model {model_name}:\n")
        log_file.write(f"Mean Squared Error (MSE): {mse}\n")
        log_file.write(f"Mean Absolute Error (MAE): {mae}\n")
        log_file.write(f"R² Score: {r2}\n")
        log_file.write(f"Relative MSE (% of avg price): {relative_mse:.2f}%\n")
        log_file.write(f"Relative MAE (% of avg price): {relative_mae:.2f}%\n")
        log_file.write("\n" + "="*50 + "\n")
    
    # return mse, mae, r2, relative_mse, relative_mae

# Function to train and evaluate the Decision Tree Regressor with different configurations
def train_and_evaluate_dt(X_train, y_train, X_test, y_test):
    # Initialize models
    dt = DecisionTreeRegressor(random_state=GT_ID)
    bagging = BaggingRegressor(estimator =dt, n_estimators=50, random_state=GT_ID)
    boosting = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=GT_ID)
    
    # GridSearchCV for tuning Decision Tree
    param_grid = {
        'max_depth': [3, 5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='neg_mean_squared_error')
    
    # Fit models
    models = {
        "Default Decision Tree": dt,
        "Tuned Decision Tree (GridSearch)": grid_search,
        "Bagging with Decision Tree": bagging,
        "Boosting with Decision Tree": boosting
    }
    
    results = {}
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"For model {model_name}:")
        evaluate_metrics_in_context(y_test, y_pred, model_name)
        plots.plot_predictions_vs_actuals(y_test, y_pred, 
                                          model_name, f"{AGGREGATED_OUTDIR}/pred_actual_diff_{model_name}.png" )
        
        # Calculate losses
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[model_name] = {
            "MSE": mse,
            "MAE": mae,
            "R2": r2
        }
    
    return results

# Function to save results to a pickle file
def save_results(results,filename ):

    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {filename}")

def check_etl():
    X, y = etl.get_data()
    df = pd.concat([X, y], axis=1)  # Concatenate X and y for easier manipulation

    #######################
    # Separate numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(exclude=['number']).columns

    # Fill missing values
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())  # Fill numerical columns with mean
    # df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])  # Fill categorical columns with mode
    for col in categorical_cols:
        if df[col].mode().empty:
            # If mode is not available (column is empty or all NaN), fill with a default value
            df[col] = df[col].fillna("Unknown")  # Or any other default value like "Missing"
        else:
            # Otherwise, fill with the mode
            try:
                df[col] = df[col].fillna(df[col].mode().iloc[0])
            except:
                print(df[col].mode())

    # Split back into X and y after dropping NaN rows
    X = df.iloc[:, :-1]  # All columns except the last one (X)
    y = df.iloc[:, -1]  # Only the last column (y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=GT_ID)
    test_data_etl_input_check(X,y,X_train, X_test, y_train, y_test, show = True)
    etl.graph_raw_data(X, y)

    print("======> Data verification complete")
    return X,y,X_train, X_test, y_train, y_test 

###############
def main(): 
    np.random.seed(GT_ID)
    X,y,X_train, X_test, y_train, y_test  = check_etl()
    check_data_info(X, y, X_train, X_test, y_train, y_test, show = False)

    ######
    # result_save_file = f"{Y_PRED_OUTDIR}/results.pkl"
    # if not os.path.exists(result_save_file):
    #     results = train_and_evaluate_dt(X_train, y_train, X_test, y_test)
    #     print("Model Evaluation Results:", results)
    #     save_results(results, f"{Y_PRED_OUTDIR}/results.pkl")
    # else:
    #     with open(result_save_file, 'rb') as f:
    #         results = pickle.load(f)
    # res_vis_png_path = f"{AGGREGATED_OUTDIR}/results.png" 
    # plots.plot_dt_results(results, res_vis_png_path)

    # nn_performance_path = f'{PERFM_PKL_OUTDIR}/perf_results.pkl'
    # cv_losses_outpath = f'{CV_LOSSES_PKL_OUTDIR}/cv_losses.pkl'
    # y_pred_outpath = f'{Y_PRED_OUTDIR}/y_pred_compare.pkl' 
    # get_eval_with_nn(X,y,nn_performance_path,cv_losses_outpath, y_pred_outpath, do_cv = 0)

    nn_performance_reg_path = f'{PERFM_PKL_OUTDIR}/perf_reg_results.pkl'
    cv_losses_reg_outpath = f'{CV_LOSSES_PKL_OUTDIR}/cv_reg_losses.pkl'
    y_pred_reg_outpath = f'{Y_PRED_OUTDIR}/y_pred_reg_compare.pkl' 
    nn_results, cv_losses, y_preds = get_eval_reg_with_nn(X,y,nn_performance_reg_path,cv_losses_reg_outpath, y_pred_reg_outpath, do_cv = 0)
 

if __name__ == "__main__":
    ###################
    print("PyTorch mps check: ",torch.backends.mps.is_available())
    print("PyTorch cuda check: ",torch.cuda.is_available())
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kf = KFold(n_splits=K_FOLD_CV, shuffle=True, random_state=GT_ID)
    print(f"Torch will be running on {device}")
    ####################
    main()
    