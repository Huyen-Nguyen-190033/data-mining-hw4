
import torch
from torch import nn
import pandas as pd 
import numpy as np
import torch.autograd as autograd


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class LSTMModel(nn.Module):
    """
    LSTM model that will be used to perform Sentiment analysis.
    """
    def __init__(self, vocab_size, output_size, embedding_dim, embedding_matrix,\
        hidden_dim, n_layers, input_len, pretrain=False):
        """
        Initialize the model by setting up the layers.
        """
        super().__init__()
        
        self.output_size = output_size  # y_out size = 1
        self.n_layers = n_layers   # layers of LSTM
        self.hidden_dim = hidden_dim  # hidden dim of LSTM
        self.input_len = input_len # len of input features
        
        ## set up pre-train embeddings. if true, load pretrain-embedding from GloVe
        if pretrain:
            # print("import glove embedding to nn.Embedding now")
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix,freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.init_weights()
        
        ## define LSTM model
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True, bidirectional=True)
        ## dropout layer
        self.dropout = nn.Dropout(0.5)
        
        ## max pool
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        ## linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim * 2, output_size)
        self.sigmoid = nn.Sigmoid()
        

    ## initialize the weights
    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
    
    ## initial hidden state and cell state
    def _init_hidden(self, batch_size):
        return(autograd.Variable(torch.randn(self.n_layers, batch_size, self.hidden_dim)).to(device),
                autograd.Variable(torch.randn(self.n_layers, batch_size, self.hidden_dim)).to(device)
                )

    
    ## feed input x into LSTM model for training/testing
    def forward(self, x):
        batch_size = x.size(0)

        embeds = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        lstm_out, _ = self.lstm(embeds)  # (batch_size, seq_len, hidden_dim)
        lstm_out = lstm_out.permute(0, 2, 1)  # (batch_size, hidden_dim, seq_len)

        pooled = self.pool(lstm_out).squeeze(2)  # (batch_size, hidden_dim)
        out = self.dropout(pooled)
        out = self.fc(out)
        #out = self.sigmoid(out)

        return out.squeeze()
        