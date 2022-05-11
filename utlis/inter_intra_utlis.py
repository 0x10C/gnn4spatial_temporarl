import torch.utils.data as torch_data
import torch
from sklearn import metrics
import torch.nn as nn
import random
import logging

import numpy as np
import scipy
import pickle
import gzip
import os
from scipy.sparse import csr_matrix

from tqdm import tqdm

class DataProcess:
    """
        return: label --> <array> #timestamp * #categroy
                feature --> <list> length = #timestamp,[#node * #feature_embedding]

    """
    def __init__(self,path):
        self.path = path

    def preprocess_label(self):
        """

        :param label_list:
        save:8598 x 5 (graph number x category number)
        :return:
        """
        ReadList = np.load(self.path['feature'], allow_pickle=True)
        a = ReadList['train_targets']
        b = ReadList['val_targets']
        rlt = np.concatenate((a, b))
        np.save(self.path['save'] + "labels.npy", rlt)
        print("label shape:{}".format(rlt.shape))
        return rlt

    def preprocess_feature(self):
        """

        save:[array<node_number x feature embedding>] length: graph number
        :return:
        """
        ReadList = np.load(self.path['feature'], allow_pickle=True)
        a = ReadList['train_feature']
        b = ReadList['val_feature']
        c = np.concatenate((a, b))
        print("feature:{}".format(c.shape[0]))

        rlt = [ele for ele in c]
        output = open(self.path['save'] + "feature_matrices.pkl", "wb")
        pickle.dump(rlt, output)

        return rlt


    def read_data(self):
        if not os.path.exists(self.path['save']):
            os.makedirs(self.path['save'])

        label = self.preprocess_label()
        print("Label finished")

        feature = self.preprocess_feature()
        print("feature matrices finished")

        return feature,label

    @staticmethod
    def align_data(feature,label,supply,window_size,save_path = None):
        len_feature = len(feature)
        len_label = label.shape[0]
        len_supple = len(supply)
        assert len_feature == len_label,"different timestamp number of label and feature!"
        assert len_supple == len_label,"different timestamp number of supplement data!"
        rlt_feature = []
        rlt_label = []
        rlt_supplement = []
        for i in range(window_size,len_feature,1):
            temp = np.array(feature[i-window_size:i+1])
            temp_supply = supply[i-window_size:i+1]
            rlt_feature.append(temp)
            rlt_label.append(label[i])
            rlt_supplement.append(temp_supply)
        # output_feature = open(save_path + "feature_matrices_align.pkl", "wb")
        # pickle.dump(rlt_feature, output_feature)
        # np.save(save_path + "labels_align.npy", rlt_label)
        print("feature align length:{}".format(len(rlt_feature)))
        return np.array(rlt_feature),rlt_label,np.array(rlt_supplement)

def normalized(data, normalize_method, norm_statistic=None):
    if normalize_method == 'min_max':
        if not norm_statistic:
            norm_statistic = dict(max=np.max(data, axis=0), min=np.min(data, axis=0))
        scale = norm_statistic['max'] - norm_statistic['min'] + 1e-5
        data = (data - norm_statistic['min']) / scale
        data = np.clip(data, 0.0, 1.0)
    elif normalize_method == 'z_score':
        if not norm_statistic:
            norm_statistic = dict(mean=np.mean(data, axis=0), std=np.std(data, axis=0))
        mean = norm_statistic['mean']
        std = norm_statistic['std']
        std = [1 if i == 0 else i for i in std]
        data = (data - mean) / std
        norm_statistic['std'] = std
    return data, norm_statistic

def model_fw_rt_batch_loss(model,criterion,train_data,label,supply):
    logits = model(torch.Tensor(train_data), supply)
    loss = criterion(logits, torch.Tensor(label))
    loss = loss.sum()
    return loss


def calculate_loss_cmp(model_att,model_avg,model_single,data_list,label_list,supply_list,batch_size, criterion):
    batch_loss_avg = 0
    batch_loss_att = 0
    batch_loss_single = 0
    for idx in range(batch_size):
        train_data = next(data_list)
        label = next(label_list)
        supply = next(supply_list)
        batch_loss_avg += model_fw_rt_batch_loss(model_avg,criterion,train_data,label,supply)
        batch_loss_att += model_fw_rt_batch_loss(model_att, criterion, train_data, label, supply)
        batch_loss_single += model_fw_rt_batch_loss(model_single, criterion, train_data, label, supply)

    return batch_loss_att / batch_size,batch_loss_avg / batch_size,batch_loss_single / batch_size


def calculate_loss(model, data_list,label_list,supply_list,adj_list,batch_size, criterion,mode):
    batch_loss = 0
    for idx in range(batch_size):
        train_data = next(data_list)
        label = next(label_list)
        supply = next(supply_list)
        adj = next(adj_list)
        adj_matrix = torch.Tensor(adj)
        if mode in ['gcn','gat']:
            logits = model(adj_matrix,torch.Tensor(train_data))
        if mode == "gps":
            node_ids, forest = supply
            logits = model(adj_matrix,node_ids,torch.Tensor(train_data))
        if mode == "sage":
            node_ids, forest = supply
            logits = model(forest,torch.Tensor(train_data))

        loss = criterion(logits, torch.Tensor(label))
        loss = loss.sum()
        batch_loss += loss
    return batch_loss / batch_size

def regression_predict(model, test_data,test_label):
    model.eval()
    test_num = len(test_data)
    error_sum = 0
    with torch.no_grad():
        for i in range(test_num):
            test = torch.Tensor(test_data[i])
            logits = model(test)
            label = test_label[i]
            error = metrics.mean_squared_error(np.array(logits),np.array(label))
            error_sum += error
    return error_sum/test_num

def acc_f1(model, test_data,test_label,supply,adj,mode):
    model.eval()
    test_num = len(test_data)
    with torch.no_grad():
        y_pred = []
        y_true = []
        for i in range(test_num):
            test = torch.Tensor(test_data[i])
            test_supply = supply[i]
            test_adj = adj[i]
            adj_matrix = torch.Tensor(test_adj)
            if mode in ['gcn', 'gat']:
                logits = model(adj_matrix, torch.Tensor(test))
            if mode == "gps":
                node_ids, forest = test_supply
                logits = model(adj_matrix, node_ids, torch.Tensor(test))
            if mode == "sage":
                node_ids, forest = test_supply
                logits = model(forest, torch.Tensor(test))
            label = test_label[i]

            y_pred.append(nn.Sigmoid()(logits).numpy())
            y_true.append(label)
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    pred = np.argmax(y_pred, axis=1)
    true = np.argmax(y_true, axis=1)

    return metrics.accuracy_score(true, pred), metrics.f1_score(true, pred, average='macro'), metrics.confusion_matrix(
        true, pred)

class ExperiDataset(torch_data.Dataset):
    def __init__(self,feature,label):
        self.feature = feature
        self.label = label

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, index):
        return self.feature[index],self.label[index]

def nonstandard_read(path,mode = "raw"):
    if mode == "re":
        ReadList = np.load(path['feature'], allow_pickle=True)
        return ReadList['feature'],ReadList['target']

    if mode == "raw":
        ReadList = np.load(path['feature'], allow_pickle=True)
        Fold_Data = ReadList['Fold_data']  # Data of each fold
        Fold_Label = ReadList['Fold_label']  # Labels of each fold
        feature = []
        label = []
        for ele in Fold_Data:
            feature.extend([e for e in ele])
        for ele in Fold_Label:
            label.extend([e for e in ele])
        return np.array(feature),np.array(label)

class CompactAdjacency:

    def __init__(self, adj, precomputed=None, subset=None):
        """Constructs CompactAdjacency.

    Args:
      adj: scipy sparse matrix containing full adjacency.
      precomputed: If given, must be a tuple (compact_adj, degrees).
        In this case, adj must be None. If supplied, subset will be ignored.
    """
        if adj is None:
            return

        if precomputed:
            if adj is not None:
                raise ValueError('Both adj and precomputed are set.')
            if subset is not None:
                logging.info('WARNING: subset is provided. It is ignored, since precomputed is supplied.')
            self.compact_adj, self.degrees = precomputed
            self.num_nodes = len(self.degrees)
        else:
            self.adj = adj
            self.num_nodes = len(self.adj) if isinstance(self.adj, dict) else self.adj.shape[0]
            self.compact_adj = scipy.sparse.dok_matrix(
                (self.num_nodes, self.num_nodes), dtype='int32')
            self.degrees = np.zeros(shape=[self.num_nodes], dtype='int32')
            self.node_set = set(subset) if subset is not None else None

            for v in range(self.num_nodes):
                if isinstance(self.adj, dict) and self.node_set is not None:
                    connection_ids = np.array(list(self.adj[v].intersection(self.node_set)))
                elif isinstance(self.adj, dict) and self.node_set is None:
                    connection_ids = np.array(list(self.adj[v]))
                else:
                    connection_ids = self.adj[v].nonzero()[1]

                self.degrees[v] = len(connection_ids)
                self.compact_adj[v, np.arange(len(connection_ids), dtype='int32')] = connection_ids

        self.compact_adj = self.compact_adj.tocsr()

    @staticmethod
    def from_file(filename):
        instance = CompactAdjacency(None, None)
        data = pickle.load(gzip.open(filename, 'rb'))
        instance.compact_adj = data['compact_adj']
        instance.adj = data['adj']
        instance.degrees = data['degrees'] if 'degrees' in data else data['lengths']
        instance.num_nodes = data['num_nodes']
        return instance

    @staticmethod
    def from_directory(directory):
        instance = CompactAdjacency(None, None)
        instance.degrees = np.load(os.path.join(directory, 'degrees.npy'))
        instance.compact_adj = scipy.sparse.load_npz(os.path.join(directory, 'cadj.npz'))
        logging.info('\n\ncompact_adj.py from_directory\n\n')
        # Make adj from cadj and save to adj.npz
        import IPython
        IPython.embed()
        instance.adj = scipy.sparse.load_npz(os.path.join(directory, 'adj.npz'))
        instance.num_nodes = instance.adj.shape[0]
        return instance

    def save(self, filename):
        with gzip.open(filename, 'wb') as fout:
            pickle.dump({
                'compact_adj': self.compact_adj,
                'adj': self.adj,
                'degrees': self.degrees,
                'num_nodes': self.num_nodes,
            }, fout)

    def neighbors_of(self, node):
        neighbors = self.compact_adj[node, :self.degrees[node]].todense()
        return np.array(neighbors)[0]

def np_uniform_sample_next(compact_adj, tree, fanout):
    last_level = tree[-1]  # [batch, f^depth]
    batch_lengths = compact_adj.degrees[last_level]
    nodes = np.repeat(last_level, fanout, axis=1)
    batch_lengths = np.repeat(batch_lengths, fanout, axis=1)
    batch_next_neighbor_ids = np.random.uniform(size=batch_lengths.shape, low=0, high=1 - 1e-9)
    # Shape = (len(nodes), neighbors_per_node)
    batch_next_neighbor_ids = np.array(
        batch_next_neighbor_ids * batch_lengths,
        dtype=last_level.dtype)
    shape = batch_next_neighbor_ids.shape
    batch_next_neighbor_ids = np.array(
        compact_adj.compact_adj[nodes.reshape(-1), batch_next_neighbor_ids.reshape(-1)]).reshape(shape)

    return batch_next_neighbor_ids

def np_traverse(compact_adj, seed_nodes, feature_emb, fanouts=(1,), sample_fn=np_uniform_sample_next):
    if not isinstance(seed_nodes, np.ndarray):
        raise ValueError('Seed must a numpy array')

    if len(seed_nodes.shape) > 2 or len(seed_nodes.shape) < 1 or not str(seed_nodes.dtype).startswith('int'):
        raise ValueError('seed_nodes must be 1D or 2D int array')

    if len(seed_nodes.shape) == 1:
        seed_nodes = np.expand_dims(seed_nodes, 1)

    # Make walk-tree
    forest_array = [seed_nodes]

    for f in fanouts:
        next_level = sample_fn(compact_adj, forest_array, f)

        assert next_level.shape[1] == forest_array[-1].shape[1] * f


        forest_array.append(next_level)

    return forest_array

def walk_forest(feature,fan_out):
    node_ids = np.array(list(range(feature.shape[0])), dtype=np.int64)
    adj = np.ones([feature.shape[0],feature.shape[0]])
    compact_adj = CompactAdjacency(csr_matrix(adj))
    forest = np_traverse(compact_adj,node_ids,feature,fan_out)
    torch_forest = [torch.from_numpy(forest[0]).flatten()]
    for i in range(len(forest) - 1):
        torch_forest.append(torch.as_tensor(torch.from_numpy(forest[i + 1]).reshape(-1, fan_out[i]),dtype = torch.int64))
    return (node_ids,torch_forest)

def adpator_sage_gps(feature,fan_out = [2,2]):
    fan_out_list = [fan_out] * len(feature)
    node_ids_forest = [walk_forest(feature[i],fan_out_list[i]) for i in tqdm(range(len(feature)))]
    return node_ids_forest

def run_model(args,args_model,path,Model):
    data_process = DataProcess(path)
    feature, label = data_process.read_data()
    # feature, label = nonstandard_read(path)

    supplement_sage_gps = adpator_sage_gps(feature)
    featrue_align, label_align,supplement_sage_gps_align = DataProcess.align_data(feature, label, supplement_sage_gps, args_model['win_size'], path['save'])

    train_index = int(len(featrue_align) * 0.75)
    shuffle = [i for i in range(len(featrue_align))]
    random.shuffle(shuffle)
    test_data = featrue_align[shuffle[train_index:]]
    test_label = label[shuffle[train_index:]]
    test_supplement_sage_gps = supplement_sage_gps_align[shuffle[train_index:]]

    model = Model(**args_model)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    if args.task == "regression":
        criterion = torch.nn.MSELoss()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    test_best_rlt = -1
    print("win size = {}".format(args_model['win_size']))

    for i in range(args.epochs):
        print("epoch:{}".format(i))
        train_data = iter(featrue_align[shuffle[:train_index]])
        train_lable = iter(label[shuffle[:train_index]])
        train_supplement_sage_gps = iter(supplement_sage_gps_align[shuffle[:train_index]])
        for b_i in range(int(train_index / args.batch_size)):
            optimizer = opt
            optimizer.zero_grad()
            batch_loss = calculate_loss(model, train_data, train_lable,train_supplement_sage_gps, args.batch_size, criterion)
            batch_loss.backward()
            optimizer.step()
            # print("batch:{} loss:{}".format(b_i,batch_loss))

            if b_i % args.frequency_of_the_test == 0 or b_i == int(train_index / args.batch_size) - 1:
                if args.task == "classification":
                    acc, f1, cm = acc_f1(model, test_data, test_label,test_supplement_sage_gps)
                    print("acc:{},f1:{}".format(acc, f1))
                    if f1 > test_best_rlt:
                        test_best_rlt = f1
                if args.task == "regression":
                    mean_square_error = regression_predict(model, test_data, test_label)
                    if mean_square_error >test_best_rlt:
                        test_best_rlt = mean_square_error
                    # print("mean_square_error:{}".format(mean_square_error))
    return test_best_rlt

