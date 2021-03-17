
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import argparse

from models.CNN import CNN
from dataloaders.ArticlesDataset import ArticlesDataset
from train_utils import save_checkpoint, save_metrics, load_pretrained_embeddings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


def train(model,
          optimizer,
          train_loader,
          valid_loader,
          save_path,
          criterion,
          num_epochs=50,
          eval_every=50,
          best_valid_loss=float("Inf"),
          model_name = "model"):

    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    print("Start training for", num_epochs, "epochs...")
    model.float()
    model.train()
    for epoch in range(num_epochs):
        print("Epoch", epoch + 1, "of", num_epochs)
        for train_batch in train_loader:
            labels = train_batch['binary_label'].unsqueeze(1).to(device)
            content = train_batch['content']
            content = torch.stack(content, dim=1).to(device)
            output = model(content).unsqueeze(1).to(device)

            loss = criterion(output, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    # validation loop
                    for val_batch in valid_loader:
                        labels = val_batch['binary_label'].unsqueeze(1).to(device)
                        content = val_batch['content']
                        content = torch.stack(content, dim=1).to(device)
                        output = model(content).unsqueeze(1).to(device)

                        loss = criterion(output, labels.float())
                        valid_running_loss += loss.item()

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0
                valid_running_loss = 0.0
                model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, global_step, num_epochs * len(train_loader),
                              average_train_loss, average_valid_loss))

                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(save_path + model_name + '.pt', model, optimizer, best_valid_loss, log_file)
                    save_metrics(save_path + model_name + '_metrics.pt', train_loss_list, valid_loss_list, global_steps_list, log_file)

    save_metrics(save_path + model_name + '_metrics.pt', train_loss_list, valid_loss_list, global_steps_list, log_file)
    print('Finished Training!')


parser = argparse.ArgumentParser(description='Training CNN Text classification')
parser.add_argument('--num_epochs', type=int, default=10, help='training epochs')
parser.add_argument('--emb_dim', type=int, default=50, help='dimension of embedding layer')
parser.add_argument('--num_kernel', type=int, default=30, help='num of each kind of kernel')
parser.add_argument('--text_len', type=int, default=100, help='text length')
parser.add_argument('--stride', type=int, default=1, help='stride')
parser.add_argument('--mlp_hidden_size', type=int, default=64, help='hidden size')
parser.add_argument('--data_path', type=str, default='data/', help='path to the train-valid-test sets')
parser.add_argument('--train_file', type=str, default='train.csv', help='filename of train set')
parser.add_argument('--valid_file', type=str, default='valid.csv', help='filename of valid set')
parser.add_argument('--vocab_file', type=str, default='vocab.pkl', help='filename of vocab dict')
parser.add_argument('--labels_file', type=str, default='label2id.pkl', help='labels dictionary')
parser.add_argument('--save_path', type=str, default='', help='path to save trained model; same as data_path by default')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
parser.add_argument('--step_size', type=int, default=1024, help='step_size')
parser.add_argument('--pretrained_emb', type=str, default='', help='path to pretrained word embeddings')
parser.add_argument('--num_runs', type=int, default=1, help='number of replicates')

args = parser.parse_args()

print("="*5, "Training multilabel CNN classifier on Reuters data", "="*5)
print("Data path:", args.data_path)
print("train_file:", args.train_file)
print("valid_file:", args.valid_file)
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

# Prepare data
train_file = data_path + args.train_file
valid_file = data_path + args.valid_file

vocab = pickle.load(open(data_path + args.vocab_file, 'rb'))
label2id = pickle.load(open(data_path + args.labels_file, 'rb'))
num_classes = len(label2id)
print("vocab:", len(vocab))
print("num_classes:", num_classes)

train_data = ArticlesDataset(csv_file=train_file, vocab=vocab, label2id=label2id, max_text_len=args.text_len)
valid_data = ArticlesDataset(csv_file=valid_file, vocab=vocab, label2id=label2id, max_text_len=args.text_len)

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)

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

for run_num in range(args.num_runs):
    print("-" * 10, "Run", run_num + 1, "-" * 10)

    # generate random seed per run
    # seed_num = 1
    # np.random.seed(seed_num)
    # torch.manual_seed(seed_num)
    # print("seed:", seed_num)

    emb_name = args.pretrained_emb.split("/")[-1][:-4]
    model_name = "cnn_" + str(args.emb_dim) + "embDim_" + \
                 str(args.num_kernel) + "kernels_" + \
                 str(args.stride) + "stride_" + \
                 str(args.mlp_hidden_size) + "MLPhidden_" + \
                 str(args.num_epochs) + "epochs" + "_" + emb_name
    model_name += "_run" + str(run_num + 1)

    print("Model name:", model_name)

    log_file = open(save_path + model_name + "_logs.txt", 'w')
    model = CNN(cnn_args=cnn_args, mlp_args=mlp_args).to(device)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()

    train(model=model,
          optimizer=optimizer,
          num_epochs=num_epochs,
          criterion=loss_fn,
          eval_every=args.step_size,
          train_loader=train_loader,
          valid_loader=valid_loader,
          save_path=save_path,
          model_name=model_name)

    print("Done training! Best model saved at", save_path + model_name + ".pt")
    log_file.close()

