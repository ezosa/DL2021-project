import torch
import torch.nn as nn
import math

from models.MLP import MLP

# CNN text classifier adapted from
# https://github.com/FernandoLpz/Text-Classification-CNN-PyTorch


class CNN(nn.ModuleList):
    def __init__(self, cnn_args, mlp_args):
        super(CNN, self).__init__()

        # embedding layer
        self.embedding_dim = cnn_args['emb_dim']
        self.embedding = nn.Embedding(cnn_args['vocab_size'], self.embedding_dim)
        # initialize with pretrained embeddings
        print("Initializing with pretrained embeddings")
        self.embedding.weight.data.copy_(cnn_args['pretrained_emb'])

        # Dropout definition
        self.dropout = nn.Dropout(0.25)

        # CNN parameters definition
        # Kernel sizes
        self.kernel_1 = 2
        self.kernel_2 = 3
        self.kernel_3 = 4
        self.kernel_4 = 5

        # Num kernels for each convolution size
        self.seq_len = cnn_args['text_len']
        # Output size for each convolution
        self.out_channels = cnn_args['num_kernel']
        # Number of strides for each convolution
        self.stride = cnn_args['stride']

        # Convolution layers definition
        self.conv_1 = nn.Conv1d(self.seq_len, self.out_channels, self.kernel_1, self.stride)
        self.conv_2 = nn.Conv1d(self.seq_len, self.out_channels, self.kernel_2, self.stride)
        self.conv_3 = nn.Conv1d(self.seq_len, self.out_channels, self.kernel_3, self.stride)
        self.conv_4 = nn.Conv1d(self.seq_len, self.out_channels, self.kernel_4, self.stride)

        # Max pooling layers definition
        self.pool_1 = nn.MaxPool1d(self.kernel_1, self.stride)
        self.pool_2 = nn.MaxPool1d(self.kernel_2, self.stride)
        self.pool_3 = nn.MaxPool1d(self.kernel_3, self.stride)
        self.pool_4 = nn.MaxPool1d(self.kernel_4, self.stride)

        # MLP classfier
        mlp_input_size = self.in_features_fc()
        #print("mlp_input_size:", mlp_input_size)
        self.mlp = MLP(input_size=mlp_input_size,
                       hidden_size=mlp_args['hidden_size'],
                       num_classes=mlp_args['num_classes'])

    def in_features_fc(self):
        '''Calculates the number of output features after Convolution + Max pooling

        Convolved_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1
        Pooled_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1

        source: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        '''
        # Calcualte size of convolved/pooled features for convolution_1/max_pooling_1 features
        out_conv_1 = ((self.embedding_dim - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
        out_conv_1 = math.floor(out_conv_1)
        out_pool_1 = ((out_conv_1 - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
        out_pool_1 = math.floor(out_pool_1)

        # Calcualte size of convolved/pooled features for convolution_2/max_pooling_2 features
        out_conv_2 = ((self.embedding_dim - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
        out_conv_2 = math.floor(out_conv_2)
        out_pool_2 = ((out_conv_2 - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
        out_pool_2 = math.floor(out_pool_2)

        # Calcualte size of convolved/pooled features for convolution_3/max_pooling_3 features
        out_conv_3 = ((self.embedding_dim - 1 * (self.kernel_3 - 1) - 1) / self.stride) + 1
        out_conv_3 = math.floor(out_conv_3)
        out_pool_3 = ((out_conv_3 - 1 * (self.kernel_3 - 1) - 1) / self.stride) + 1
        out_pool_3 = math.floor(out_pool_3)

        # Calcualte size of convolved/pooled features for convolution_4/max_pooling_4 features
        out_conv_4 = ((self.embedding_dim - 1 * (self.kernel_4 - 1) - 1) / self.stride) + 1
        out_conv_4 = math.floor(out_conv_4)
        out_pool_4 = ((out_conv_4 - 1 * (self.kernel_4 - 1) - 1) / self.stride) + 1
        out_pool_4 = math.floor(out_pool_4)

        # Returns "flattened" vector (input for fully connected layer)
        return (out_pool_1 + out_pool_2 + out_pool_3 + out_pool_4) * self.out_channels

    def forward(self, x):

        # Sequence of tokes is filtered through an embedding layer
        x = self.embedding(x)

        # Convolution layer 1 is applied
        x1 = self.conv_1(x)
        x1 = torch.relu(x1)
        x1 = self.pool_1(x1)

        # Convolution layer 2 is applied
        x2 = self.conv_2(x)
        x2 = torch.relu((x2))
        x2 = self.pool_2(x2)

        # Convolution layer 3 is applied
        x3 = self.conv_3(x)
        x3 = torch.relu(x3)
        x3 = self.pool_3(x3)

        # Convolution layer 4 is applied
        x4 = self.conv_4(x)
        x4 = torch.relu(x4)
        x4 = self.pool_4(x4)

        # The output of each convolutional layer is concatenated into a unique vector
        union = torch.cat((x1, x2, x3, x4), 2)
        union = union.reshape(union.size(0), -1)

        #print("union:", union.shape)
        # The "flattened" vector is passed to an MLP classifier
        mlp_output = self.mlp(union)

        return mlp_output
