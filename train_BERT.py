# finetuning BERT for text classification
# code adapted from https://github.com/dipansh-girdhar/Text-classification-for-long-text/

from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch

import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from textwrap import wrap
from sklearn.metrics import f1_score, recall_score, precision_score

from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataloaders.BERTArticlesDataset import BERTArticlesDataset

data_path = "./bin/DL2021-project/data/"
save_path = "./results/DL2021_project/"

PRETRAINED_BERT = 'bert-base-cased'
EPOCHS = 3
MAX_LEN = 256
BATCH_SIZE = 16
HIDDEN_SIZE = 64
NUM_CLASSES = 103

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


def defining_bert_tokenizer(PRETRAINED_BERT):
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_BERT)
    return tokenizer


def create_data_loader(df, label2id, tokenizer, max_len, batch_size):
    ds = BERTArticlesDataset(
        df=df,
        label2id=label2id,
        tokenizer=tokenizer,
        max_len=max_len)

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4)


class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRETRAINED_BERT)
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        #print("input_ids:", input_ids.shape)
        #print("attention_mask:", attention_mask.shape)
        input_ids = input_ids.squeeze(1)
        attention_mask = attention_mask.squeeze(1)
        #print("input_ids:", input_ids.shape)
        #print("attention_mask:", attention_mask.shape)

        output_dict = self.bert(
            return_dict=True,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        #print("outputs:", output_dict.keys())
        pooled_output = output_dict['pooler_output']
        #print("pooled_output:")
        #print(pooled_output)
        output = self.drop(pooled_output)
        output = self.fc(output)
        output = self.sigmoid(output)
        #print("output:", output.shape)
        #print(output)
        return output


def train_epoch(
        model,
        train_loader,
        loss_fn,
        optimizer,
        scheduler):

    model.float()
    model.train()
    losses = []

    for train_batch in train_loader:
        input_ids = train_batch["input_ids"].to(device)
        attention_mask = train_batch["attention_mask"].to(device)
        targets = train_batch["targets"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # append new loss
        loss = loss_fn(outputs, targets.float())
        losses.append(loss.item())

        # compute accuracy
        # _, preds = torch.max(outputs, dim=1)
        # correct_predictions += torch.sum(preds == targets)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return np.mean(losses)


def eval_model(model, data_loader, loss_fn):
    model = model.eval()
    losses = []

    with torch.no_grad():
        for data_batch in data_loader:
            input_ids = data_batch["input_ids"].to(device)
            attention_mask = data_batch["attention_mask"].to(device)
            targets = data_batch["targets"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # append new loss
            loss = loss_fn(outputs, targets)
            losses.append(loss.item())

            # compute accuracy
            # _, preds = torch.max(outputs, dim=1)
            # correct_predictions += torch.sum(preds == targets)

    return np.mean(losses)


def get_predictions(model, data_loader, threshold=0.5):
    review_texts = []
    prediction_probs = []
    real_values = []
    predictions = []
    targets = []

    model = model.eval()
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets_i = d["targets"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask)

            # pred = torch.max(outputs, dim=1)
            # probs = F.softmax(outputs, dim=1)
            preds_i = (outputs > threshold).int()

            print("targets:", targets_i.shape)
            print("predictions:", preds_i.shape)
            predictions.extend(preds_i.tolist())
            targets.extend(targets_i.squeeze(1).tolist())

            # predictions.extend(preds)
            # prediction_probs.extend(probs)
            # real_values.extend(targets)

    y_true = np.array(targets)
    y_pred = np.array(predictions)
    print("mean y_true labels:", np.mean(y_true, axis=1))
    print("mean y_pred labels:", np.mean(y_pred, axis=1))
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print("macro-F1:", macro_f1)
    recall = recall_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    print("macro Recall:", recall)
    print("macro Precision:", precision)

    results = {'f1': macro_f1,
               'recall': recall,
               'precision': precision}

    return results


if __name__ == "__main__":

    print("BERT Model:", PRETRAINED_BERT)
    print("Loading pretrained BERT...")
    bert_model = BertModel.from_pretrained(PRETRAINED_BERT)
    tokenizer = defining_bert_tokenizer(PRETRAINED_BERT)

    label2id = pickle.load(open(data_path + "label2id.pkl", 'rb'))
    df_train = pd.read_csv(data_path + "train.csv")
    df_valid = pd.read_csv(data_path + "valid.csv")
    df_test = pd.read_csv(data_path + "test.csv")

    print("train data:", df_train.shape)
    print("valid data:", df_valid.shape)
    print("test data:", df_test.shape)

    train_loader = create_data_loader(df_train, label2id, tokenizer, MAX_LEN, BATCH_SIZE)
    valid_loader = create_data_loader(df_valid, label2id, tokenizer, MAX_LEN, BATCH_SIZE)
    test_loader = create_data_loader(df_test, label2id, tokenizer, MAX_LEN, BATCH_SIZE)

    data = next(iter(train_loader))

    model = Classifier(NUM_CLASSES).to(device)

    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    loss_fn = nn.BCELoss().to(device)

    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(EPOCHS):

        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            scheduler
        )

        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_loss = eval_model(
            model,
            valid_loader,
            loss_fn
        )

        print(f'Val   loss {val_loss} accuracy {val_acc}')

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), save_path + 'best_BERT_model_state.bin')
            best_accuracy = val_acc

    print("Finished finetuning BERT! Saved model at", save_path + "best_BERT_model_state.bin")

    test_loss = eval_model(
        model,
        test_loader,
        loss_fn
    )
    print('\nTest loss:\n')
    print(test_loss.item())

    results = get_predictions(
        model,
        test_loader
    )

    # print(classification_report(y_test, y_pred, target_names=class_names))


# PRETRAINED_BERT = 'bert-base-cased'
#
# tokenizer = defining_bert_tokenizer(PRETRAINED_BERT)
# sample_txt = 'Personal Health Record (Extract)\nCreated on October 24, 2019\nPatient\nSteven Fuerst\nBirthdate\nDecember 10, 1979\nRace\nInformation not\navailable'
# tokens = tokenizer.tokenize(sample_txt)
#
# encoding = tokenizer.encode_plus(
#                       sample_txt,
#                       max_length=64,
#                       add_special_tokens=True, # Add '[CLS]' and '[SEP]'
#                       return_token_type_ids=False,
#                       pad_to_max_length=True,
#                       return_attention_mask=True,
#                       return_tensors='pt',  # Return PyTorch tensors
#                     )
#
# bert_model = BertModel.from_pretrained(PRETRAINED_BERT)
#
# last_hidden_state, pooled_output = bert_model(
#                     return_dict=True,
#                     input_ids=encoding['input_ids'],
#                     attention_mask=encoding['attention_mask'])
