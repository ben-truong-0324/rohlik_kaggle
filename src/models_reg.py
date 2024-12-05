

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



class MLPRegression(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, dropout_rate=0.01):
        super(MLPRegression, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 10)
        self.fc4 = nn.Linear(hidden_dim // 3, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        return self.fc4(x)  # Output a continuous value
    
class MLPRegression_lessRelu(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, dropout_rate=0.01):
        super(MLPRegression, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 10)
        self.fc4 = nn.Linear(hidden_dim // 3, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        # x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        return self.fc4(x)  # Output a continuous value

class CNNRegression(nn.Module):
    def __init__(self, input_channels, output_dim, conv_filters=64, kernel_size=3, dropout_rate=0.1):
        super(CNNRegression, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, conv_filters, kernel_size, padding=1)
        self.conv2 = nn.Conv1d(conv_filters, conv_filters * 2, kernel_size, padding=1)
        self.fc = nn.Linear(conv_filters * 2, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.dropout(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)  # Pool to reduce spatial dimensions
        return self.fc(x)
    

class LSTMRegression(nn.Module):
    def __init__(self, input_dim, output_dim, lstm_hidden_dim=128, dropout_rate=0.1):
        super(LSTMRegression, self).__init__()
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = torch.relu(lstm_out[:, -1, :])  # Take the output from the last time step
        x = self.dropout(x)
        return self.fc(x)  # Output continuous value
    

class ConvLSTMRegression(nn.Module):
    def __init__(self, input_dim, output_dim, conv_hidden_dim=64, lstm_hidden_dim=128, dropout_rate=0.1):
        super(ConvLSTMRegression, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, conv_hidden_dim, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(conv_hidden_dim, lstm_hidden_dim, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = torch.relu(self.conv1(x.transpose(1, 2)))  # Conv1D expects (batch, channels, sequence)
        x = x.transpose(1, 2)  # Switch back to (batch, sequence, channels)
        lstm_out, _ = self.lstm(x)
        x = torch.relu(lstm_out[:, -1, :])  # Take the output from the last time step
        x = self.dropout(x)
        return self.fc(x)  # Output continuous value