import torch
import torch.nn as nn
import numpy as np


class MLP_score(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MLP_score, self).__init__()
        self.activation_1 = nn.ReLU()
        self.layer1 = nn.Linear(in_channel, 256)
        self.layer2 = nn.Linear(256, 64)
        self.layer3 = nn.Linear(64, out_channel)

    def forward(self, x):
        x = self.activation_1(self.layer1(x))
        x = self.activation_1(self.layer2(x))
        output = self.layer3(x)
        return output


class LogitScaleNetwork(nn.Module):
    def __init__(self, init_scale=2):
        super(LogitScaleNetwork, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(init_scale))  # from openclip

    def forward(self, x=None):  # add x to make it compatible with DDP
        return self.logit_scale.exp()

class ScoreNet(nn.Module):
    def __init__(self, in_channel):
        super(ScoreNet, self).__init__()


        self.activation_1 = nn.ReLU()
        self.layer1 = nn.Linear(in_channel, 256)
        self.layer2 = nn.Linear(256, 64)
        self.layer3 = nn.Linear(64, 1)

        # self.activation_1 = nn.ReLU()
        # self.layer1 = nn.Linear(in_channel, 256)
        # self.layer2 = nn.Linear(256, 64)
        # self.layer3 = nn.Linear(64, 16)
        # self.layer4 = nn.Linear(16, 4)
        # self.layer5 = nn.Linear(4, 1)

        self.seg2all2 = nn.Linear(3, 1)


    def forward(self, input_feature):


        x = input_feature
        batch_size = x.shape[0]
        dim = s.shape[-1]
        s0 = x[:, :5, :].unsqueeze(-1)
        s1 = x[:, 5:10, :].unsqueeze(-1)
        s2 = x[:, 10:, :].unsqueeze(-1)
        x = torch.cat((s0, s1, s2), -1)  # bs,5,368,3
        x = x.permute(0, 3, 1, 2).reshape(batch_size, 3, 5 * dim)


        x = self.activation_1(self.layer1(x))
        x = self.activation_1(self.layer2(x))
        output = self.layer3(x)
        # print(output.size())  # 16，3，1
        output = output.reshape(output.shape[0], -1)


        output = self.seg2all2(output).reshape(output.shape[0], -1)

        # x = self.activation_1(self.layer1(x))
        # x = self.activation_1(self.layer2(x))
        # x = self.activation_1(self.layer3(x))
        # x = self.activation_1(self.layer4(x))
        # output = self.layer5(x)

        '''
        s0 = self.seg0(x[:, :5])
        s1 = self.seg1(x[:, 5:10])
        s2 = self.seg2(x[:, 10:15])
        s = torch.cat((s0, s1, s2), -1)
        # s = torch.sigmoid(s)
        # s = s / torch.sum(s, dim=1, keepdim=True)
        output = self.activation_1(self.out(s))
        '''
        return output


