import einops
import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    def __init__(self, mlp_dim: int, hidden_dim: int, dropout=0.):
        super(MLPBlock, self).__init__()
        self.mlp_dim = mlp_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.Linear1 = nn.Linear(mlp_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.Linear2 = nn.Linear(hidden_dim, mlp_dim)

    def forward(self, x):
            x = self.Linear1(x)
            x = self.gelu(x)
            x = self.dropout(x)
            x = self.Linear2(x)
            x = self.dropout(x)
            return x


class Mixer_struc(nn.Module):
    def __init__(self, patches: int, token_dim: int, dim: int, channel_dim: int, dropout=0.):
        super(Mixer_struc, self).__init__()
        self.patches = patches
        self.channel_dim = channel_dim
        self.token_dim = token_dim
        self.dropout = dropout

        self.MLP_block_token = MLPBlock(token_dim, dim, self.dropout)
        self.MLP_block_chan = MLPBlock(channel_dim, dim, self.dropout)
        self.LayerNorm = nn.LayerNorm(dim)

    def forward(self, x): #(?, 256(h*w), 512(c))
        out = self.LayerNorm(x)
        out = einops.rearrange(out, 'b n d -> b d n') # (?, 512(c), 256(h*w)) 对 h*w 进行 linear 操作
        out = self.MLP_block_token(out)
        out = einops.rearrange(out, 'b d n -> b n d') # (?, 256(h*w), 512(c)) 对channel 进行 linear 操作
        out += x
        out2 = self.LayerNorm(out)
        out2 = self.MLP_block_chan(out2)
        out2 += out
        return out2


class MLP_Mixer(nn.Module):
    def __init__(self, image_size, patch_size, token_dim, channel_dim, num_classes, dim, num_blocks):
        super(MLP_Mixer, self).__init__()
        # n_patches = (image_size // patch_size) ** 2
        n_patches = (image_size // patch_size)
        # self.patch_size_embbeder = nn.Conv2d(kernel_size=n_patches, stride=n_patches, in_channels=3, out_channels=dim)
        # self.patch_size_embbeder = nn.Conv2d(kernel_size=patch_size, stride=patch_size, in_channels=3, out_channels=channel_dim)
        self.patch_size_embbeder = nn.Conv1d(kernel_size=patch_size, stride=patch_size, in_channels=1,
                                                out_channels=channel_dim)
        self.blocks = nn.ModuleList([
            Mixer_struc(patches=n_patches, token_dim=token_dim, channel_dim=channel_dim, dim=dim) for i in
            range(num_blocks)
        ])

        self.Layernorm1 = nn.LayerNorm(dim)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=dim * 10, out_features=num_classes),
            nn.Softmax(),
        )
        # self.classifier = nn.Sequential(
        # nn.Linear(in_features=dim * 10, out_features=64),
        # nn.Dropout(0.5),
        # nn.Linear(in_features=64, out_features=num_classes),
        # nn.Softmax(), # 配合BECLoss
        # )

    def forward(self, x):
        out = torch.unsqueeze(x, dim=-1) # (?, 10,3000,1 )
        out = einops.rearrange(out, "n c h w -> (n c) w h")
        out = self.patch_size_embbeder(out) # (?, 512, 16, 16) (?*10, 512, 30)(n, c,seq)
        # out = einops.rearrange(out, "n c h w -> n (h w) c") # (?, 256, 512)
        out = einops.rearrange(out, "n c seq -> n seq c")
        for block in self.blocks:
            out = block(out)
        out = self.Layernorm1(out)
        out = out.mean(dim=1) # out（n_sample,dim）
        out = out.contiguous().view(x.size()[0], x.size()[1], -1)
        result = self.classifier(out.view(out.size()[0], -1))
        return result, out


if __name__ == "__main__":
    x = torch.randn(64, 10, 3000)

    # x = torch.randn(64, 3, 256, 256)

    # from mlp_mixer_pytorch import MLPMixer
    #
    # model = MLPMixer(
    # image_size=(256, 128),
    # channels=3,
    # patch_size=16,
    # dim=512,
    # depth=12,
    # num_classes=5
    # )
    #
    # img = torch.randn(1, 3, 256, 128)
    # pred = model(img) # (1, 1000)

    print("_______")

    model = MLP_Mixer(image_size=3000, patch_size=100, dim=256, num_classes=5, num_blocks=8, token_dim=30,
    channel_dim=256)

    model(x)