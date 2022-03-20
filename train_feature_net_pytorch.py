import torch
import argparse
import os
import shutil
import numpy as np
import logging
from utlis.Utils import *
from torch.utils.data import Dataset, DataLoader
from MlpMixerNet import MLP_Mixer
from tqdm import  tqdm
logging.basicConfig(level=logging.DEBUG,
format='%(asctime)s %(filename)s %(levelname)s %(message)s',
datefmt='%a, %d %b %Y %H:%M:%S',
filename='train_feature_net_pytorch_covlstm.log',
filemode='w')

logger = logging.getLogger()


class kFoldGenerator():
    '''
    Data Generator
    '''
    k = -1      # the fold number
    x_list = [] # x list with length=k
    y_list = [] # x list with length=k

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


def train(feature_net, optimizer, loss_f, train_data, train_targets, batch_size_f, epoch):
    feature_net.train()

    train_dataset = feature_dataset(train_data, train_targets)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size_f, shuffle=True)
    train_losses = []
    num_correct = 0
    for i, (data, label) in enumerate(train_dataloader):
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


def valid(feature_net, loss_f, val_data, val_targets, batch_size_f, epoch):
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


def generate_feature(feature_net, train_data, val_data):
    if hasattr(torch.cuda, "empty_cache"):
        torch.cuda.empty_cache()

    train_dataset = feature_dataset(train_data, train_targets)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size_f, shuffle=True)

    val_dataset = feature_dataset(val_data, val_targets)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size_f, shuffle=False)

    feature_net.eval()
    train_features = np.array([])
    val_features = np.array([])
    with torch.no_grad():
        for i, (data, label) in enumerate(train_dataloader):
            data = data.to(torch.float32).to(device)
            _, train_feature = feature_net(data) # train_feature shape (batch_size, 10 , 256)
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


device = "cpu"

if __name__ == "__main__":

    print(128 * '#')
    print('Start to train FeatureNet.')

    # command line parameters -c -g
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=str, help="configuration file", default="./config/ISRUC.config")
    parser.add_argument("-g", type=str, help="GPU number to use, set '-1' to use CPU", default=-1)
    args = parser.parse_args()
    args.c = "./config/ISRUC.config"
    args.g = "0"
    # Path, cfgFeature, _, _ = ReadConfig(args.c)
    Path, cfgFeature,_ = ReadConfig(args.c)

    if args.g != "-1":
        device = torch.device("cuda:" + args.g if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"

    print(args.g, device)

    # ## 1.2. Analytic parameters

    # [train] parameters ('_f' means FeatureNet)
    channels = int(cfgFeature["channels"])
    fold = int(cfgFeature["fold"])
    num_epochs_f = int(cfgFeature["epoch_f"])
    batch_size_f = int(cfgFeature["batch_size_f"])
    optimizer_f = cfgFeature["optimizer_f"]
    learn_rate_f = float(cfgFeature["learn_rate_f"])

    # ## 1.3. Parameter check and enable

    # Create save pathand copy .config to it
    if not os.path.exists(Path['Save']):
        os.makedirs(Path['Save'])
    shutil.copyfile(args.c, Path['Save'] + "last.config")

    # # 2. Read data and process data

    # ## 2.1. Read data
    # Each fold corresponds to one subject's data (ISRUC-S3 dataset)
    ReadList = np.load(Path['data'], allow_pickle=True)
    Fold_Num = ReadList['Fold_len'] # Num of samples of each fold [924,911,...]
    Fold_Data = ReadList['Fold_data'] # Data of each fold
    Fold_Label = ReadList['Fold_label'] # Labels of each fold

    print("Read data successfully")
    print('Number of samples: ', np.sum(Fold_Num))

    # ## 2.2. Build kFoldGenerator or DominGenerator
    # 只是把data和label读出来了
    DataGenerator = kFoldGenerator(Fold_Data, Fold_Label)
    save_best_acc_feature_net_dir = "./output/feature_net/convlstm"
    # # 3. Model training (cross validation)

    # k-fold cross validation
    all_scores = []
    for i in range(0, fold):
        print(128 * '_')
        print('Fold #', i)


        feature_net = MLP_Mixer(image_size=3000, patch_size=100, dim=256, num_classes=5, num_blocks=6, token_dim=30,
        channel_dim=256).to(device)
        optimizer = get_optimizer(feature_net, optimizer_f, learn_rate_f)
        # loss_f = torch.nn.CrossEntropyLoss()
        loss_f = torch.nn.BCELoss()

        # get i th-fold data
        train_data, train_targets, val_data, val_targets = DataGenerator.getFold(i)

        best_acc = 0
        for epoch in tqdm(range(num_epochs_f)):
            train(feature_net, optimizer, loss_f, train_data, train_targets, batch_size_f, epoch)
            val_acc = valid(feature_net, loss_f, val_data, val_targets, batch_size_f, epoch)

            if best_acc < val_acc:
                best_acc = val_acc
            if not os.path.exists(save_best_acc_feature_net_dir):
                os.makedirs(save_best_acc_feature_net_dir)
            save_best_acc_feature_net_path = os.path.join(save_best_acc_feature_net_dir,
                "feature_net_best_acc_{}.pkl".format(i))
            logger.info("save feature_net_{} best acc:{} ...".format(i, best_acc))
            torch.save(feature_net.state_dict(), save_best_acc_feature_net_path)

            feature_net.load_state_dict(torch.load(os.path.join(save_best_acc_feature_net_dir,
            "feature_net_best_acc_{}.pkl".format(i))))

            train_features, val_features = generate_feature(feature_net, train_data, val_data)

            logger.info(
            ('Save feature of Fold #' + str(i) + ' to' + Path['Save'] + 'Feature_' + str(
            i) + '_pytorch_convlstm.npz'))

            np.savez(Path['Save'] + 'Feature_' + str(i) + '_new.npz', # 保存 网络输出的node特征和对应的label
            train_feature=train_features, # train_feature: (924, 10, 256)
            val_feature=val_features,
            train_targets=train_targets, # train_targets: (924, 5)
            val_targets=val_targets
            )

        logger.info(128 * '_')

    logger.info('End of training FeatureNet.')
    logger.info(128 * '#')