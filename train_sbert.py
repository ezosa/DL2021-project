import pandas as pd
import numpy as np
import time
import pickle
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

from train_utils import save_checkpoint, save_metrics
from dataloaders.CustomLoader import CustomLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

LABEL2ID = pickle.load(open("datafiles/labels2id.pkl","rb"))
MODEL_NAME = "model-bceloss"
LOG_FILE = f"train-{MODEL_NAME}.{int(time.time())}.log"

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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden_activation=F.relu):
        x = hidden_activation(self.fc1(x))
        x = self.dropout(x)
        x = hidden_activation(self.fc2(x))
        x = self.dropout(x)
        return self.sigmoid(self.fc3(x))

def drop_col(c):
    try:
        unnamed = [x for x in c.columns if "nnamed" in x]
        c = c.drop(columns=unnamed)
    except KeyError:
        pass
    return c

def convert_codes(row):
    global LABEL2ID
    if pd.isna(row):
        return np.zeros(len(LABEL2ID))
    else:
        row = row.split()
        codes2id = np.array([LABEL2ID[code] for code in row if code in LABEL2ID])
        binary_label = np.zeros(len(LABEL2ID))
        binary_label[codes2id] = 1
        
    return binary_label

def log(info):
    with open(LOG_FILE, "a") as f:
        f.write(info+"\n")

def read_dataset(dfname):
    now = time.time()
    print(f"[*] Reading {dfname}...")
    df = drop_col(pd.read_csv(dfname))
    X = torch.Tensor(df.drop(columns=["codes"]).to_numpy())
    y = torch.Tensor(df['codes'].apply(convert_codes))
    print(f"[!] Loaded {dfname}, took {time.time() - now:.2f}s")
    return X, y

def plotloss(train_loss_list, val_loss_list):
    sns.set()
    plt.plot(train_loss_list)
    plt.plot(val_loss_list)
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(LOG_FILE.replace("log","png"))

def train(model,
          optimizer,
          train_loader,
          valid_loader,
          save_path,
          criterion,
          num_epochs=50,
          eval_every_batches=50,
          best_valid_loss=float("Inf"),
          model_name = "model"):

    # initialize running values
    now = time.time()
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []
    save_path = save_path if save_path.endswith("/") else save_path+"/"

    # training loop
    try:
        print("Start training for", num_epochs, "epochs...")
        model.to(DEVICE)
        model.float()
        model.train()
        for epoch in range(num_epochs):
            print("Epoch", epoch + 1, "of", num_epochs)
            for train_batch in train_loader:
                labels = train_batch[1].to(DEVICE)
                content = train_batch[0].to(DEVICE)
                output = model(content).to(DEVICE)
                # labels & output shapes: [128, 103]

                loss = criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # update running values
                running_loss += loss.item()
                global_step += 1

                # evaluation step
                if global_step % eval_every_batches == 0:
                    model.eval()
                    with torch.no_grad():
                        # validation loop
                        for val_batch in valid_loader:
                            labels = val_batch[1].to(DEVICE)
                            content = val_batch[0].to(DEVICE)
                            output = model(content).to(DEVICE)

                            loss = criterion(output, labels)
                            valid_running_loss += loss.item()

                            # DOESN'T WORK
                            #macro_f1 = f1_score(labels.to_list(), output.to_list(), average='macro')
                            #log(f"macro-F1: {macro_f1}")
                            #log(f"{labels.shape}, {type(labels), {output.shape}, type(output)}")

                    # evaluation
                    average_train_loss = running_loss / eval_every_batches
                    average_valid_loss = valid_running_loss / len(valid_loader)
                    train_loss_list.append(average_train_loss)
                    valid_loss_list.append(average_valid_loss)
                    global_steps_list.append(global_step)

                    # resetting running values
                    running_loss = 0.0
                    valid_running_loss = 0.0
                    model.train()

                    # print progress
                    printline = 'Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}, Time: {:.2f}s'\
                        .format(epoch + 1, num_epochs, global_step, num_epochs * len(train_loader),\
                                average_train_loss, average_valid_loss, time.time() - now)
                    print(printline)
                    log(printline)
                    now = time.time()

                    # checkpoint
                    if best_valid_loss > average_valid_loss:
                        best_valid_loss = average_valid_loss
                        save_checkpoint(save_path + model_name + '.pt', model, optimizer, best_valid_loss, open(LOG_FILE, "a"))
                        save_metrics(save_path + model_name + '_metrics.pt', train_loss_list, valid_loss_list, global_steps_list, open(LOG_FILE, "a"))
    
    except KeyboardInterrupt:
        pass
    
    # [5:] to avoid huge losses of first 5 measurement and "hockey stick" graph
    plotloss(train_loss_list[5:], valid_loss_list[5:])

    save_metrics(save_path + model_name + '_metrics.pt', train_loss_list, valid_loss_list, global_steps_list, open(LOG_FILE, "a"))
    print('Finished Training!')


if __name__ == "__main__":
    start = time.time()

    train_XT, train_yT = read_dataset("datafiles/train_enc.csv") # ~30 s
    val_XT, val_yT = read_dataset("datafiles/val_enc.csv")
    #test_XT, test_yT = read_dataset("datafiles/test_enc.csv")

    EMBEDDING_DIM = val_XT.shape[1]
    OUTPUT_DIM = val_yT.shape[1]
    BATCH_SIZE = 128

    model = FFNN(EMBEDDING_DIM, OUTPUT_DIM)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    train_custom_loader = CustomLoader(train_XT, train_yT)
    val_custom_loader = CustomLoader(val_XT, val_yT)

    train_loader = DataLoader(train_custom_loader, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(val_custom_loader, batch_size=BATCH_SIZE, shuffle=True)

    train(model, optimizer, train_loader, valid_loader, "datafiles", criterion, num_epochs=100, model_name=MODEL_NAME)
    print(f"Execution time: {time.time() - start:.2f}s", file=open(LOG_FILE, "a"))
