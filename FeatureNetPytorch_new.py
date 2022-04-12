import torch


class smallFilterCNN(torch.nn.Module):
    def __init__(self, channels=10):
        super(smallFilterCNN, self).__init__()
        self.conv_layer0 = torch.nn.Conv1d(in_channels=1, out_channels=32, kernel_size=50, stride=6)
        self.batch_norm_layer0 = torch.nn.BatchNorm1d(32)
        self.relu_layer0 = torch.nn.ReLU(inplace=True)
        self.maxpool_layer0 = torch.nn.MaxPool1d(kernel_size=16, stride=16)
        self.dropout_layer0 = torch.nn.Dropout(p=0.5)

        self.conv_layer1 = torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8, stride=1, padding="same")
        self.batch_norm_layer1 = torch.nn.BatchNorm1d(64)
        self.relu_layer1 = torch.nn.ReLU(inplace=True)

        self.conv_layer2 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=8, stride=1, padding="same")
        self.batch_norm_layer2 = torch.nn.BatchNorm1d(64)
        self.relu_layer2 = torch.nn.ReLU(inplace=True)

        self.conv_layer3 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=8, stride=1, padding="same")
        self.batch_norm_layer3 = torch.nn.BatchNorm1d(64)
        self.relu_layer3 = torch.nn.ReLU(inplace=True)

        self.conv_layer4 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=8, stride=1, padding="same")
        self.batch_norm_layer4 = torch.nn.BatchNorm1d(64)
        self.relu_layer4 = torch.nn.ReLU(inplace=True)

        self.maxpool_layer1 = torch.nn.MaxPool1d(kernel_size=8, stride=8)

    def forward(self, x):
        # x = self.relu_layer0(self.batch_norm_layer0(self.conv_layer0(x.to(torch.float32))))
        x = self.relu_layer0(self.batch_norm_layer0(self.conv_layer0(x)))
        # (640,32,492)
        x = self.maxpool_layer0(x)
        # (640,,32,30)
        x = self.dropout_layer0(x)

        x = self.relu_layer1(self.batch_norm_layer1(self.conv_layer1(x)))
        x = self.relu_layer2(self.batch_norm_layer2(self.conv_layer2(x)))
        x = self.relu_layer3(self.batch_norm_layer3(self.conv_layer3(x)))
        x = self.relu_layer4(self.batch_norm_layer4(self.conv_layer4(x)))

        y_pred = self.maxpool_layer1(x)

        return y_pred.view(-1, y_pred.size()[1] * y_pred.size()[2])


class largeFilterCNN(torch.nn.Module):
    def __init__(self, channels=10):
        super(largeFilterCNN, self).__init__()
        self.conv_layer0 = torch.nn.Conv1d(in_channels=1, out_channels=64, kernel_size=400, stride=50)
        self.batch_norm_layer0 = torch.nn.BatchNorm1d(64)
        self.relu_layer0 = torch.nn.ReLU(inplace=True)
        self.maxpool_layer0 = torch.nn.MaxPool1d(kernel_size=8, stride=8)
        self.dropout_layer0 = torch.nn.Dropout(p=0.5)

        self.conv_layer1 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=6, stride=1, padding="same")
        self.batch_norm_layer1 = torch.nn.BatchNorm1d(64)
        self.relu_layer1 = torch.nn.ReLU(inplace=True)

        self.conv_layer2 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=6, stride=1, padding="same")
        self.batch_norm_layer2 = torch.nn.BatchNorm1d(64)
        self.relu_layer2 = torch.nn.ReLU(inplace=True)

        self.conv_layer3 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=6, stride=1, padding="same")
        self.batch_norm_layer3 = torch.nn.BatchNorm1d(64)
        self.relu_layer3 = torch.nn.ReLU(inplace=True)

        self.conv_layer4 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=6, stride=1, padding="same")
        self.batch_norm_layer4 = torch.nn.BatchNorm1d(64)
        self.relu_layer4 = torch.nn.ReLU(inplace=True)

        self.maxpool_layer1 = torch.nn.MaxPool1d(kernel_size=4, stride=4)

    def forward(self, x):
        # x = self.relu_layer0(self.batch_norm_layer0(self.conv_layer0(x.to(torch.float32))))
        x = self.relu_layer0(self.batch_norm_layer0(self.conv_layer0(x)))
        x = self.maxpool_layer0(x)
        x = self.dropout_layer0(x)

        x = self.relu_layer1(self.batch_norm_layer1(self.conv_layer1(x)))
        x = self.relu_layer2(self.batch_norm_layer2(self.conv_layer2(x)))
        x = self.relu_layer3(self.batch_norm_layer3(self.conv_layer3(x)))
        x = self.relu_layer4(self.batch_norm_layer4(self.conv_layer4(x)))

        y_pred = self.maxpool_layer1(x)

        return y_pred.view(-1, y_pred.size()[1] * y_pred.size()[2])


class TimeDistributed(torch.nn.Module): # stackoverflow上的pytorch的TimeDistributed的实现
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-2), x.size(-1)).permute(0, 2, 1) # (samples * timesteps,1,input_size)
        y = self.module(x_reshape) # x_reshape (640,3000,1)
        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1)) # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1)) # (timesteps, samples, output_size)

        return y


class featureNet(torch.nn.Module): # input shape = (?, 10, 3000)
    def __init__(self, channels=10, num_cat=5):
        super(featureNet, self).__init__()
        self.small_filter_model = smallFilterCNN(channels=channels)
        self.large_filter_model = largeFilterCNN(channels=channels)

        self.time_distributed_small = TimeDistributed(self.small_filter_model)
        self.time_distributed_large = TimeDistributed(self.large_filter_model)

        self.linears = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features=channels*256, out_features=64),
            torch.nn.Linear(in_features=64, out_features=num_cat),
            torch.nn.Softmax(), # 配合BECLoss
        )

    def forward(self, x):
        # x = x.type(torch.cuda.FloatTensor)
        x = torch.unsqueeze(x, -1) # (batch_size, 10, 3000) --> (batch_size, 10, 3000, 1)

        # for i in range(x.size()[1]): # 模拟keras的TimeDistributed
        # x_c = x[:, i, :, :]
        # small_filter_feature = self.small_filter_model(x_c)
        # large_filter_feature = self.large_filter_model(x_c)
        # concat_feature = torch.cat((small_filter_feature, large_filter_feature), dim=1)
        # x[:, i] = concat_feature

        x_small_feature = self.time_distributed_small(x) # (?, timesteps, output_size)
        x_large_feature = self.time_distributed_large(x)

        x = torch.cat((x_small_feature, x_large_feature), dim=-1)

        x_flatten = x.view(-1, x.size()[1] * x.size()[2]) # node feature
        y_pred = self.linears(x_flatten)

        # y_pred = torch.log(y_pred) # 与 交叉熵 保持一致

        return y_pred, x