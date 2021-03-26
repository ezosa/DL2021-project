import pickle
import torch
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, recall_score, precision_score

from train_utils import load_checkpoint, load_metrics
from dataloaders.CustomLoader import CustomLoader
from train_sbert import read_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

LABEL2ID = pickle.load(open("datafiles/labels2id.pkl","rb"))
MODEL_NAME = "model-bce-deep"
LOG_FILE = f"test-{MODEL_NAME}.{int(time.time())}.log"

class FFNN(nn.Module):
    # Feel free to add whichever arguments you like here.
    def __init__(self, sbert_dim, n_classes, hidden_neurons_1=256, hidden_neurons_2=128, dropout_rate=0.5):
        # super(FFNN, self).__init__() # obsolete syntax
        super().__init__()
        # WRITE CODE HERE
        self.fc1 = nn.Linear(sbert_dim, hidden_neurons_1)
        self.fc2 = nn.Linear(hidden_neurons_1, hidden_neurons_2)
        self.fc3 = nn.Linear(hidden_neurons_2, n_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, hidden_activation=F.relu):
        x = hidden_activation(self.fc1(x))
        x = self.dropout(x)
        x = hidden_activation(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

class FFNN_DEEP(nn.Module):
    # Feel free to add whichever arguments you like here.
    def __init__(self, sbert_dim, n_classes, hidden_neurons_1=512, hidden_neurons_2=256, dropout_rate=0.5):
        # super(FFNN, self).__init__() # obsolete syntax
        super().__init__()
        self.fc1 = nn.Linear(sbert_dim, hidden_neurons_1)
        self.fc2 = nn.Linear(hidden_neurons_1, hidden_neurons_1)
        self.fc3 = nn.Linear(hidden_neurons_1, hidden_neurons_2)
        self.fc4 = nn.Linear(hidden_neurons_2, hidden_neurons_2)
        self.fc5 = nn.Linear(hidden_neurons_2, n_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden_activation=F.relu):
        x = hidden_activation(self.fc1(x))
        x = self.dropout(x)
        x = hidden_activation(self.fc2(x))
        x = self.dropout(x)
        x = hidden_activation(self.fc3(x))
        x = self.dropout(x)
        x = hidden_activation(self.fc4(x))
        x = self.dropout(x)
        return self.sigmoid(self.fc5(x))


def log(info):
    with open(LOG_FILE, "a") as f:
        f.write(info+"\n")

def evaluate(model, test_loader, threshold=0.5):
    y_pred = []
    y_true = []

    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        for i, test_batch in enumerate(test_loader):
            
            labels = test_batch[1].to(DEVICE)
            content = test_batch[0].to(DEVICE)
            
            y_score_i = model(content).to(DEVICE)
            y_pred_i = (y_score_i > threshold).int()
            
            #logdata = f"[{i+1}] Predictions: shape {y_score_i.shape}, prob_tensor:\n{y_score_i},\nlabel tensor:\n {y_pred_i}"
            #log(logdata)

            y_pred.extend(y_pred_i.tolist())
            y_true.extend(labels.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    log(f"mean y_true labels: {np.mean(y_true, axis=1)}")
    log(f"mean y_pred labels: {np.mean(y_pred, axis=1)}")
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    log(f"macro-F1: {macro_f1}")
    recall = recall_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    log(f"macro Recall: {recall}")
    log(f"macro Precision: {precision}")

    results = {'f1': macro_f1,
               'recall': recall,
               'precision': precision}

    return results

if __name__ == "__main__":

    #val_XT, val_yT = read_dataset("datafiles/val_enc.csv")
    test_XT, test_yT = read_dataset("datafiles/test_enc.csv")

    EMBEDDING_DIM = test_XT.shape[1]
    OUTPUT_DIM = test_yT.shape[1]
    BATCH_SIZE = 128

    #model = FFNN(EMBEDDING_DIM, OUTPUT_DIM)
    model = FFNN_DEEP(EMBEDDING_DIM, OUTPUT_DIM)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    test_custom_loader = CustomLoader(test_XT, test_yT)
    test_loader = DataLoader(test_custom_loader, batch_size=BATCH_SIZE, shuffle=True)

    load_checkpoint(f"datafiles/{MODEL_NAME}.pt", model, optimizer, DEVICE, open(LOG_FILE, "a"))
    results = evaluate(model, test_loader)
    #log(results)