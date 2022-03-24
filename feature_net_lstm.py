import torch



class featureNetLSTM(torch.nn.Module):
    def __init__(self, input_size=100, hidden_size=256, num_layers=1,
            batch_first=True, dropout=0.8, bidirectional=True, proj_size=0):
        super(featureNetLSTM, self).__init__()

        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.proj_size = proj_size

        # self.small_filter_model = smallFilterCNN(channels=10)
        # self.large_filter_model = largeFilterCNN(channels=10)
        #
        # self.time_distributed_small = TimeDistributed(self.small_filter_model)
        # self.time_distributed_large = TimeDistributed(self.large_filter_model)

        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                    batch_first=batch_first, dropout=dropout, bidirectional=bidirectional,
                                    proj_size=proj_size)

        # self.feature_layer = torch.nn.Linear(in_features=30*256, out_features=256)
        self.linears = torch.nn.Sequential(
            # torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features=10*256, out_features=64),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features=64, out_features=5),
            torch.nn.Softmax(), # 配合BECLoss
        )

    def forward(self, x): # (?, 10, 3000)
        # x_reshape = x.to(torch.float32).contiguous().view(-1, 30, 100) # (?, 10, 3000) --> (?*10, 30, 100)
        x_reshape = x.contiguous().view(-1, 30, 100)
        # x_ = torch.unsqueeze(x, -1)
        #
        # x_small_feature = self.time_distributed_small(x_) # (?, timesteps, output_size)
        # x_large_feature = self.time_distributed_large(x_)
        # fea = torch.cat((x_small_feature, x_large_feature), dim=-1)

        h0 = torch.randn(self.num_layers*2 if self.bidirectional else self.num_layers, x_reshape.size()[0],
                        self.hidden_size if self.proj_size == 0 else self.proj_size).type(torch.FloatTensor)
        c0 = torch.randn(self.num_layers*2 if self.bidirectional else self.num_layers,
                        x_reshape.size()[0], self.hidden_size).type(torch.FloatTensor)
        output, (hn, cn) = self.lstm(x_reshape, (h0, c0)) # output: (?*10, 30, 256)

        # x_fea = output.contiguous().view(output.size()[0], -1)
        # x_fea = self.feature_layer(x_fea)

        # x_fea = output[:, -1, :].view(x.size()[0], 10, -1)
        x_fea = torch.cat((output[:, -1, :128].contiguous().view(x.size()[0], 10, -1),
        output[:, 0, -128:].contiguous().view(x.size()[0], 10, -1)), dim=-1) # x_fea: (?, 10, 256)

        # x_fea = torch.cat((x_fea, fea), dim=-1)
        x_flatten = x_fea.contiguous().view(x.size()[0], -1) # x_flatten: (?, 2560)
        y_pred = self.linears(x_flatten)


        return y_pred, x_fea


if __name__ == "__main__":

    feature_net_lstm = featureNetLSTM()

    # 原始数据： （?*10)*30*6000 -(lstm)-> ?*10*512 -(feature_cnn)-> ?*10*256