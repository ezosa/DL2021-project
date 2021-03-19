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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

LABEL2ID = pickle.load(open("datafiles/labels2id.pkl","rb"))
LOG_FILE = open(f"run.{int(time.time())}.log","w")

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

def drop_col(c):
    try:
        c = c.drop(columns=["Unnamed: 0"])
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

def read_dataset(dfname):
    now = time.time()
    print(f"[*] Reading {dfname}...")
    df = drop_col(pd.read_csv(dfname))
    X = torch.Tensor(df.drop(columns=["codes"]).to_numpy())
    y = torch.Tensor(df['codes'].apply(convert_codes))
    print(f"[!] Loaded {dfname}, took {time.time() - now:.2f}s")
    return X, y

def args():
    parser = argparse.ArgumentParser(description='Testing CNN Text classification')
    parser.add_argument('--num_epochs', type=int, default=10, help='training epochs')
    parser.add_argument('--emb_dim', type=int, default=50, help='dimension of embedding layer')
    parser.add_argument('--num_kernel', type=int, default=30, help='num of each kind of kernel')
    parser.add_argument('--text_len', type=int, default=100, help='text length')
    parser.add_argument('--stride', type=int, default=1, help='stride')
    parser.add_argument('--mlp_hidden_size', type=int, default=64, help='hidden size')
    parser.add_argument('--data_path', type=str, default='data/', help='path to the train-valid-test sets')
    parser.add_argument('--test_file', type=str, default='test.csv', help='filename of test set')
    parser.add_argument('--vocab_file', type=str, default='vocab.pkl', help='filename of vocab dict')
    parser.add_argument('--labels_file', type=str, default='label2id.pkl', help='labels dictionary')
    parser.add_argument('--save_path', type=str, default='', help='path to save trained model; same as data_path by default')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--step_size', type=int, default=1024, help='step_size')
    parser.add_argument('--pretrained_emb', type=str, default='', help='path to pretrained word embeddings')
    parser.add_argument('--num_runs', type=int, default=1, help='number of replicates')

    args = parser.parse_args()
    return args

def plotloss(train_loss_list, val_loss_list):
    sns.set()
    plt.plot(train_loss_list)
    plt.plot(val_loss_list)
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

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
        model.float()
        model.train()
        for epoch in range(num_epochs):
            print("Epoch", epoch + 1, "of", num_epochs)
            for train_batch in train_loader:
                labels = train_batch[1].to(device)
                content = train_batch[0].to(device)
                output = model(content).to(device)
                # labels & output shapes: [128, 103]

                loss = criterion(output, torch.max(labels, 1)[1])
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
                            labels = val_batch[1].to(device)
                            content = val_batch[0].to(device)
                            output = model(content).to(device)

                            loss = criterion(output, torch.max(labels,1)[1])
                            valid_running_loss += loss.item()

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
                    print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}, Took: {:.2f}s'
                        .format(epoch + 1, num_epochs, global_step, num_epochs * len(train_loader),
                                average_train_loss, average_valid_loss, time.time() - now))
                    now = time.time()

                    # checkpoint
                    if best_valid_loss > average_valid_loss:
                        best_valid_loss = average_valid_loss
                        save_checkpoint(save_path + model_name + '.pt', model, optimizer, best_valid_loss, LOG_FILE)
                        save_metrics(save_path + model_name + '_metrics.pt', train_loss_list, valid_loss_list, global_steps_list, LOG_FILE)
        
    except KeyboardInterrupt:
        pass    
    
    plotloss(train_loss_list[5:], valid_loss_list[5:])

    save_metrics(save_path + model_name + '_metrics.pt', train_loss_list, valid_loss_list, global_steps_list, LOG_FILE)
    print('Finished Training!')


if __name__ == "__main__":

    train_XT, train_yT = read_dataset("datafiles/train_enc.csv") # ~30 s
    val_XT, val_yT = read_dataset("datafiles/val_enc.csv")
    #test_XT, testXT = read_dataset("datafiles/test_enc.csv")

    EMBEDDING_DIM = val_XT.shape[1]
    OUTPUT_DIM = val_yT.shape[1]
    BATCH_SIZE = 128

    model = FFNN(EMBEDDING_DIM, OUTPUT_DIM)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_custom_loader = CustomLoader(train_XT, train_yT)
    val_custom_loader = CustomLoader(val_XT, val_yT)

    train_loader = DataLoader(train_custom_loader, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(val_custom_loader, batch_size=BATCH_SIZE, shuffle=True)

    train(model, optimizer, train_loader, valid_loader, "datafiles", criterion, num_epochs=150)