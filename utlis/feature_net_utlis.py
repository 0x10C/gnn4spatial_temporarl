import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import configparser


def ReadConfig(configfile):
    config = configparser.ConfigParser()
    print('Config: ', configfile)
    config.read(configfile)
    cfgPath = config['path']
    cfgFeat = config['feature']
    cfgTrain = config['train']
    # cfgModel = config['model']
    return cfgPath, cfgFeat, cfgTrain


class kFoldGenerator():
    '''
    Data Generator
    '''
    k = -1  # the fold number
    x_list = []  # x list with length=k
    y_list = []  # x list with length=k

    # Initializate
    def __init__(self, x, y):
        if len(x) != len(y):
            assert False, 'Data generator: Length of x or y is not equal to k.'
        self.k = len(x)
        self.x_list = x
        self.y_list = y

    # Get i-th fold
    def getFold(self, i):
        isFirst = True
        for p in range(self.k):
            if p != i:
                if isFirst:
                    train_data = self.x_list[p]
                    train_targets = self.y_list[p]
                    isFirst = False
                else:
                    train_data = np.concatenate((train_data, self.x_list[p]))
                    train_targets = np.concatenate((train_targets, self.y_list[p]))
            else:
                val_data = self.x_list[p]
                val_targets = self.y_list[p]
        return train_data, train_targets, val_data, val_targets

    # Get all data x
    def getX(self):
        All_X = self.x_list[0]
        for i in range(1, self.k):
            All_X = np.append(All_X, self.x_list[i], axis=0)
        return All_X

    # Get all label y (one-hot)
    def getY(self):
        All_Y = self.y_list[0][2:-2]
        for i in range(1, self.k):
            All_Y = np.append(All_Y, self.y_list[i][2:-2], axis=0)
        return All_Y

    # Get all label y (int)
    def getY_int(self):
        All_Y = self.getY()
        return np.argmax(All_Y, axis=1)


class feature_dataset(Dataset):
    def __init__(self, datas, labels):
        super(feature_dataset, self).__init__()
        self.datas = datas
        self.labels = labels

    def __getitem__(self, index):
        return self.datas[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


def get_optimizer(model, optimizer_f, learn_rate_f):
    if optimizer_f.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=learn_rate_f)
    elif optimizer_f == "RMSprop":
        return torch.optim.RMSprop(model.parameters(), lr=learn_rate_f)
    elif optimizer_f == "SGD":
        return torch.optim.SGD(model.parameters(), lr=learn_rate_f, momentum=0.9)
    else:
        assert False, 'Config: check optimizer, may be not implemented.'


def train(feature_net, optimizer, loss_f, train_data, train_targets, batch_size_f, epoch, device, logger):
    feature_net.train()

    train_dataset = feature_dataset(train_data, train_targets)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size_f, shuffle=True)
    train_losses = []
    num_correct = 0
    for i, (data, label) in tqdm(enumerate(train_dataloader)):
        data = data.to(torch.float32).to(device)
        label = label.to(torch.float32).to(device)
        y_pred, _ = feature_net(data)
        # train_loss = loss_f(y_pred, label.to(torch.float32))
        train_loss = loss_f(y_pred, label)
        train_losses.append(train_loss.item())

        num_correct += torch.eq(y_pred.argmax(dim=1), label.argmax(dim=1)).sum().item()

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    train_acc = num_correct / len(train_dataloader.dataset)
    train_loss = np.mean(np.array(train_losses))
    logger.info("epoch:{}, train_acc:{}, train_loss:{} ".format(epoch, train_acc, train_loss))
    print("epoch:{}, train_acc:{}, train_loss:{} ".format(epoch, train_acc, train_loss))
    return train_loss


def valid(feature_net, loss_f, val_data, val_targets, batch_size_f, epoch, device, logger):
    feature_net.eval()
    val_dataset = feature_dataset(val_data, val_targets)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size_f, shuffle=False)
    val_losses = []
    num_correct = 0
    with torch.no_grad():
        for i, (data, label) in enumerate(val_dataloader):
            data = data.to(torch.float32).to(device)
            label = label.to(torch.float32).to(device)
            y_pred, _ = feature_net(data)
            # val_loss = loss_f(y_pred, label.to(torch.float32))
            val_loss = loss_f(y_pred, label)
            val_losses.append(val_loss.item())
            num_correct += torch.eq(y_pred.argmax(dim=1), label.argmax(dim=1)).sum().item()

        val_acc = num_correct / len(val_dataloader.dataset)
        val_loss = np.mean(np.array(val_losses))
    logger.info("epoch:{}, val_acc:{}, val_loss:{} ".format(epoch, val_acc, val_loss))
    print("epoch:{}, val_acc:{}, val_loss:{} ".format(epoch, val_acc, val_loss))
    return val_acc


def generate_feature(feature_net, train_data, val_data, train_targets, val_targets, batch_size_f, device):
    if hasattr(torch.cuda, "empty_cache"):
        torch.cuda.empty_cache()

    train_dataset = feature_dataset(train_data, train_targets)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size_f, shuffle=False)

    val_dataset = feature_dataset(val_data, val_targets)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size_f, shuffle=False)

    feature_net.eval()
    train_features = np.array([])
    val_features = np.array([])
    with torch.no_grad():
        for i, (data, label) in enumerate(train_dataloader):
            data = data.to(torch.float32).to(device)
            _, train_feature = feature_net(data)  # train_feature shape (batch_size, 10 , 256)
            if i == 0:
                train_features = train_feature.cpu().detach().numpy()
            else:
                train_features = np.concatenate((train_features, train_feature.cpu().detach().numpy()), axis=0)

    for i, (data, label) in enumerate(val_dataloader):
        data = data.to(torch.float32).to(device)
        _, val_feature = feature_net(data)
        if i == 0:
            val_features = val_feature.cpu().detach().numpy()
        else:
            val_features = np.concatenate((val_features, val_feature.cpu().detach().numpy()), axis=0)

    return train_features, val_features
