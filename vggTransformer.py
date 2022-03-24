import torch


class vggTransformer(torch.nn.Module):
    def __init__(self):
        super(vggTransformer, self).__init__()

        self.vgg = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=32, kernel_size=450, stride=10, padding=4), # 大核
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(in_channels=32, out_channels=1, kernel_size=8, stride=1, padding="same"),
            torch.nn.BatchNorm1d(1),
            torch.nn.ReLU(inplace=True)
        )

        self.transformer = torch.nn.Transformer(d_model=256, nhead=8, num_encoder_layers=2, num_decoder_layers=2,
                dim_feedforward=1024, dropout=0.5, activation="relu", batch_first=True,
                layer_norm_eps=1e-05)

        self.linears = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features=256*10, out_features=64),
            torch.nn.Linear(in_features=64, out_features=5),
            torch.nn.Softmax(), # 配合BECLoss
        )

    def forward(self, x): # (?, 10, 3000)
        x_fea = torch.unsqueeze(x.contiguous().view(-1, x.size(-1)), dim=1) # (?, 10, 3000) --> (?*10, 3000) --> (?*10, 1, 3000)
        x_fea = self.vgg(x_fea)
        x_fea = self.transformer(x_fea, x_fea)

        x_flatten = x_fea.view(-1, 10*x_fea.size()[-1]) # node feature
        y_pred = self.linears(x_flatten)

        return y_pred, x_fea.view(-1, 10, x_fea.size()[-1])