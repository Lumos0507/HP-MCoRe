import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SOS_token = 0

class Encoder(nn.Module):
    def __init__(self, in_features, hidden_size):
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.encoder = nn.LSTM(input_size=in_features, hidden_size=hidden_size, dropout=0.5, num_layers=1)  # encoder

    def forward(self, enc_input):
        seq_len, batch_size, embedding_size = enc_input.size()
        h_0 = torch.rand(1, batch_size, self.hidden_size).to(device)
        c_0 = torch.rand(1, batch_size, self.hidden_size).to(device)
        # en_ht:[num_layers * num_directions,Batch_size,hidden_size]
        encode_output, (encode_ht, decode_ht) = self.encoder(enc_input, (h_0, c_0))
        return encode_output, (encode_ht, decode_ht)


class Decoder(nn.Module):
    def __init__(self, in_features, hidden_size):
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.crition = nn.CrossEntropyLoss()
        self.fc = nn.Linear(hidden_size, in_features)
        self.decoder = nn.LSTM(input_size=in_features, hidden_size=hidden_size, dropout=0.5, num_layers=1)  # encoder

    def forward(self, enc_output, dec_input):
        (h0, c0) = enc_output
        # en_ht:[num_layers * num_directions,Batch_size,hidden_size]
        de_output, (_, _) = self.decoder(dec_input, (h0, c0))
        return de_output


class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder, in_features, hidden_size):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.fc = nn.Linear(hidden_size, in_features)
        self.crition = nn.CrossEntropyLoss()
        self.seg2all =nn.Linear(3,1)

    def forward(self, enc_input, dec_input):
        enc_input = enc_input.permute(1, 0, 2)  # [seq_len,Batch_size,embedding_size] 3,16,368*5
        dec_input = dec_input.permute(1, 0, 2)  # [seq_len,Batch_size,embedding_size]
        # output:[seq_len,Batch_size,hidden_size]
        encode_output, (ht, ct) = self.encoder(enc_input)  # en_ht:[num_layers * num_directions,Batch_size,hidden_size]
        print(encode_output.size())
        # de_output = self.decoder((ht, ct), dec_input)  # de_output:[seq_len,Batch_size,in_features]
        # output = self.fc(de_output)   # Batch_size,seq_len,hidden_size
        # output = self.seg2all(output.permute(0, 2, 1)).reshape(output.shape[0],-1)
        output = self.seg2all(encode_output.permute(0, 2, 1)).reshape(encode_output.shape[0], -1)

        return output



class MLP_score(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MLP_score, self).__init__()
        self.activation_1 = nn.ReLU()
        self.layer1 = nn.Linear(in_channel, 256)
        self.layer2 = nn.Linear(256, 64)
        self.layer3 = nn.Linear(64, out_channel)

    def forward(self, x):
        # Batch_size,
        x = self.activation_1(self.layer1(x))
        x = self.activation_1(self.layer2(x))
        output = self.layer3(x)
        return output


class EnDeScoreNet(nn.Module):
    def __init__(self,in_channel):
        super(EnDeScoreNet, self).__init__()
        hidden_size = 2048
        self.encoder = Encoder(in_channel, hidden_size)
        self.decoder = Decoder(in_channel, hidden_size)
        self.seq2seq = Seq2seq(self.encoder , self.decoder , in_channel, hidden_size)

        # self.mlp = MLP_score(in_channel,1)
        self.mlp = MLP_score(hidden_size, 1)


    def forward(self, x):
        # Batch_size,5*3,368
        batch_size = x.shape[0]
        s0 = x[:, :5,:].unsqueeze(-1)
        s1 = x[:, 5:10, :].unsqueeze(-1)
        s2 = x[:, 10:, :].unsqueeze(-1)
        x = torch.cat((s0,s1,s2),-1)  #bs,5,368,3
        x = x.permute(0,3,1,2).reshape(batch_size,3,5*368).permute(1,0,2)

        # x = x.reshape(batch_size,5,3,368)
        # x = x.permute(0,2,1,3).reshape(batch_size,3,5*368).permute(1,0,2)
        input_tensor = x
        # print(input_tensor.size())
        enc_input = input_tensor  # [seq_len,Batch_size,embedding_size] 3,16,368*5
        dec_input = input_tensor

        x = self.seq2seq(enc_input, dec_input)  # enc_input, dec_input
        # print(x.size())

        output = self.mlp(x)
        # print(output.size())

        return output