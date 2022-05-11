from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from new_model.VAE import VAE, Generator_global_distribution
import matplotlib.pyplot as plt
import numpy as np

device = "cpu"

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function_vae(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD

def loss_function_generation(recon_x, x):
    MSE = torch.sum((recon_x - x).pow(2))
    return MSE

def train(epoch, model, optimizer, train_loader, device, log_interval):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss, BCE ,KLD = loss_function_vae(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tBCE: {:.4f}\tKLD: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data),
                BCE.item() / len(data),
                KLD.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch, model, test_loader, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss,_,_ = loss_function_vae(recon_batch, data, mu, logvar)
            test_loss += loss.item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

def run_vae(clients_data,clients_valid):
    n_clients = len(clients_data)
    batch_size = 64
    epochs = 80
    cuda = torch.cuda.is_available()
    seed = 1
    log_interval = 10

    torch.manual_seed(seed)

    device = torch.device("cuda" if cuda else "cpu")
    models = []
    for i in range(n_clients):
        model = VAE().to(device)
        optimizer = optim.Adam(model.parameters(), lr=5e-4)
        kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
        train_loader = torch.utils.data.DataLoader(clients_data[i], batch_size=batch_size, shuffle=True, **kwargs)
        valid_loader = torch.utils.data.DataLoader(clients_valid[i], batch_size=batch_size, shuffle=True, **kwargs)

        for epoch in range(1, epochs + 1):
            train(epoch, model, optimizer, train_loader, device, log_interval)
            test(epoch, model, valid_loader, device)
        print()
        models.append(model)
    for i, VAE_model in enumerate(models):
        torch.save(VAE_model, "/log/VAE/Model/VAE_Client{}.pth.tar".format(i))


def draw_vae():
    logfile = ""
    for line in open("/log/VAE/vae_loss.log"):
        logfile += line

    client_log = []

    for loss_text in logfile.split("\n\n")[:5]:
        BCE_list = []
        KLD_list = []
        train_loss_list = []
        train_step_loss_list = []
        valid_step_loss_list = []

        for epoch_text in loss_text.split("Train Epoch:")[1:]:
            BCE = float(epoch_text.split("\tBCE: ")[1].split("\tKLD")[0])
            KLD = float(epoch_text.split("\tKLD: ")[1].split("\n")[0])
            train_loss = BCE + KLD
            BCE_list.append(BCE)
            KLD_list.append(KLD)
            train_loss_list.append(train_loss)

        for step_text in loss_text.split("Average loss: ")[1:]:
            train_step_loss = float(step_text.split("\n")[0])
            valid_step_loss = float(step_text.split("Test set loss: ")[1].split("\n")[0])
            train_step_loss_list.append(train_step_loss)
            valid_step_loss_list.append(valid_step_loss)

        client_log.append((BCE_list, KLD_list, train_loss_list, train_step_loss_list, valid_step_loss_list))

        plt.figure(figsize=(5, 4), dpi=200)
        for i in range(5):
            BCE_list, KLD_list, train_loss_list, train_step_loss_list, valid_step_loss_list = client_log[i]
            plt.plot(train_step_loss_list, label='Client ' + str(i))
            # plt.plot(valid_step_loss_list, label='Client Valid'+str(i))

        plt.legend()
        plt.title("VAE train loss for each client")
        plt.savefig("log/image/VAE/VAE_train_step_loss.png")
        plt.show()

        plt.figure(figsize=(5, 4), dpi=200)
        for i in range(5):
            BCE_list, KLD_list, train_loss_list, train_step_loss_list, valid_step_loss_list = client_log[i]
            plt.plot(valid_step_loss_list, label='Client ' + str(i))
            # plt.plot(valid_step_loss_list, label='Client Valid'+str(i))

        plt.legend()
        plt.title("VAE test loss for each client")
        plt.savefig("log/image/VAE/VAE_test_step_loss.png")
        plt.show()

        plt.figure(figsize=(5, 4), dpi=200)
        for i in range(5):
            BCE_list, KLD_list, train_loss_list, train_step_loss_list, valid_step_loss_list = client_log[i]

            KLD_step = []
            for j, KLD in enumerate(KLD_list):
                if j % int(len(KLD_list) / len(train_step_loss_list)) == int(
                        len(KLD_list) / len(train_step_loss_list)) - 1:
                    KLD_step.append(KLD)

            plt.plot(KLD_step, label='Client ' + str(i))
            # plt.plot(valid_step_loss_list, label='Client Valid'+str(i))
            # break
        plt.legend()
        plt.title("VAE KLD for each client")
        plt.savefig("log/image/VAE/KLD.png")
        plt.show()

        plt.figure(figsize=(5, 4), dpi=200)
        for i in range(5):
            BCE_list, KLD_list, train_loss_list, train_step_loss_list, valid_step_loss_list = client_log[i]

            BCE_step = []
            for j, BCE in enumerate(BCE_list):
                if j % int(len(BCE_list) / len(train_step_loss_list)) == int(
                        len(BCE_list) / len(train_step_loss_list)) - 1:
                    BCE_step.append(BCE)

            plt.plot(BCE_step, label='Client ' + str(i))
            # plt.plot(valid_step_loss_list, label='Client Valid'+str(i))
            # break
        plt.legend()
        plt.title("VAE BCE for each client")
        plt.savefig("log/image/VAE/BCE.png")
        plt.show()

        fig, ax = plt.subplots(figsize=(5, 4), dpi=200)

        for i in range(5):
            BCE_list, KLD_list, train_loss_list, train_step_loss_list, valid_step_loss_list = client_log[i]

            train_step_loss_max_list = []
            train_step_loss_min_list = []
            train_step_loss = []
            for j, loss in enumerate(train_loss_list):
                train_step_loss.append(loss)
                if j % int(len(train_loss_list) / len(train_step_loss_list)) == int(
                        len(train_loss_list) / len(train_step_loss_list)) - 1:
                    train_step_loss_max_list.append(min([max(train_step_loss), 700]))
                    train_step_loss_min_list.append(min(train_step_loss))
                    train_step_loss = []

            ax.plot(train_step_loss_list, label='Client ' + str(i))

            ax.fill_between(np.arange(len(train_step_loss_list)), train_step_loss_max_list, train_step_loss_min_list,
                            alpha=.5, linewidth=2)

            break

        # ax.figure(figsize=(5, 4), dpi=200)
        ax.set(xlim=(0, 50), ylim=(300, 350),
               title="VAE train loss for client 0")

        # plt.ylim(300,350)

        # fig.legend()
        # fig.title("VAE train loss for client 0")
        plt.savefig("log/image/VAE/train_step_loss_client0.png")
        plt.show()

def norm_distribution(mu,logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return

def get_data_feature(model_path, dataloader):
    load_model=torch.load(model_path)

    temp = []

    def get_feature_repre():
        def hook(model, input, output):
            temp.append(torch.sigmoid(output).detach())
        return hook

    load_model.fc21.register_forward_hook(get_feature_repre())
    load_model.fc22.register_forward_hook(get_feature_repre())

    feature_repre = []
    load_model.eval()
    for batch_idx, data in enumerate(dataloader):
        data = data.to(device)
        _,_,_ = load_model(data)

        mu, logvar = temp[0], temp[1]

        # 64个加起来的
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        feature_repre.append(sum(mu + eps*std))
        temp = []

    sum(feature_repre)

    # feature = sum([sum(item) for item in feature_repre])/sum([len(item) for item in feature_repre])

    return sum(feature_repre)/dataloader.dataset.shape[0]



