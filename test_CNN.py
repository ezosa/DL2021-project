
import pickle
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, recall_score, precision_score
import os
import json
import numpy as np
import argparse

from models.CNN import CNN
from dataloaders.ArticlesDataset import ArticlesDataset
from train_utils import load_checkpoint, load_pretrained_embeddings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


def evaluate(model, test_loader, threshold=0.5):
    y_pred = []
    y_true = []


    model.eval()
    with torch.no_grad():
        for i, test_batch in enumerate(test_loader):
            print("Batch", i+1)
            labels = test_batch['binary_label'].unsqueeze(1).to(device)
            content = test_batch['content']
            content = torch.stack(content, dim=1).to(device)
            y_score_i = model(content).to(device)

            y_pred_i = (y_score_i > threshold).int()
            #print("predictions:")
            #print(y_pred_i)
            print("y_pred_i:", y_pred_i.shape)

            y_pred.extend(y_pred_i.tolist())
            y_true.extend(labels.squeeze(1).tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
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

print("="*5, "Testing multilabel CNN classifier on Reuters data", "="*5)
print("Data path:", args.data_path)
print("test_file:", args.test_file)
print("vocab:", args.vocab_file)
print("labels:", args.labels_file)
print("save_path:", args.save_path)
print("epochs:", args.num_epochs)
print("-"*20)
print("emb_dim:", args.emb_dim)
print("num_kernel:", args.num_kernel)
print("text_len:", args.text_len)
print("stride:", args.stride)
print("mlp_hidden_size:", args.mlp_hidden_size)
print("-"*20)
print("batch_size:", args.batch_size)
print("step_size:", args.step_size)
print("pretrained_emb:", args.pretrained_emb)
print("="*60)

data_path = args.data_path
num_epochs = args.num_epochs
save_path = args.save_path

vocab = pickle.load(open(data_path + args.vocab_file, 'rb'))
label2id = pickle.load(open(data_path + args.labels_file, 'rb'))
num_classes = len(label2id)
print("vocab:", len(vocab))
print("num_classes:", num_classes)

cnn_args = {}
# load pretrained embeddings
word_emb = load_pretrained_embeddings(args.pretrained_emb, vocab, embedding_dim=args.emb_dim)
cnn_args['pretrained_emb'] = word_emb
cnn_args['vocab_size'] = len(vocab)
cnn_args['emb_dim'] = args.emb_dim
cnn_args['num_kernel'] = args.num_kernel
cnn_args['text_len'] = args.text_len
cnn_args['stride'] = args.stride

mlp_args = {}
mlp_args['hidden_size'] = args.mlp_hidden_size
mlp_args['num_classes'] = num_classes


# Prepare each test set and evaluate them for every run of the model
print("test_file:", args.test_file)

emb_name = args.pretrained_emb.split("/")[-1][:-4]
model_name = "cnn_" + str(args.emb_dim) + "embDim_" + \
             str(args.num_kernel) + "kernels_" + \
             str(args.stride) + "stride_" + \
             str(args.mlp_hidden_size) + "MLPhidden_" + \
             str(args.num_epochs) + "epochs" + "_" + emb_name

log_file = open(save_path + model_name + "_test_logs.txt", 'a+')

# prepare test loader for the test set
test_file = args.data_path + args.test_file
test_data = ArticlesDataset(csv_file=test_file, vocab=vocab, label2id=label2id, max_text_len=args.text_len)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

scores_dict = {'f1': [], 'recall': [], 'precision': [], 'confidence': []}

for run_num in range(args.num_runs):
    model_run_name = model_name + "_run"+str(run_num+1)
    print("-"*10, "Run", run_num+1, "-"*10)
    print("Model name:", model_run_name)
    print("Loading model from", save_path + model_run_name + ".pt")

    best_model = CNN(cnn_args=cnn_args, mlp_args=mlp_args).to(device)
    optimizer = torch.optim.Adam(best_model.parameters(), lr=0.005)
    load_checkpoint(save_path + model_run_name + ".pt", best_model, optimizer, device, log_file)

    results = evaluate(best_model, test_loader)
    scores_dict['f1'].append(results['f1'])
    scores_dict['recall'].append(results['recall'])
    scores_dict['precision'].append(results['precision'])

    # if args.save_confidence is True:
    #     scores_dict['confidence'].append(results['confidence'])
    #     scores_dict['labels'].append(results['labels'])
    #     scores_dict['content'].append(results['content'])
    #     sentence_encodings = results['sentence_encodings']


scores_filename = save_path + model_name + "_test_scores.json"
scores_file = open(scores_filename, 'w')
json.dump(scores_dict, scores_file)
scores_file.close()

print("\nDone! Saved test scores to", scores_filename, "!")