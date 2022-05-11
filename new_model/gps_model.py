import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


class EncoderAggregate1(nn.Module):
    """
    Policy Driven Sampling model
    """

    def __init__(self, feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(EncoderAggregate1, self).__init__()

        self.feat_dim = feat_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2

        self.layer1 = nn.Linear(2 * feat_dim, hidden_dim1, bias=False)

        torch.nn.init.xavier_uniform_(self.layer1.weight)

        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, seed_nodes, feature_matrix, adj_layer1, feature_layer1):
        feat_0 = feature_matrix[seed_nodes]  # Of shape torch.Size([|B|, feat_dim])
        # num_neigh = adj_layer1.sum(1, keepdim=True)
        # adj_layer1 = adj_layer1.div(num_neigh)

        aggregated_0 = torch.matmul(adj_layer1, feature_layer1)

        aggregated_0 = aggregated_0 / 2

        # Depth 1
        feat_0 = torch.cat((feat_0, aggregated_0), dim=1)  # Of shape torch.size(|B|, 2 * feat_dim)
        feat_0 = self.relu(self.layer1(feat_0))  # Of shape torch.size(|B|, hidden_dim1)
        feat_0 = self.dropout(feat_0)

        return feat_0


class EncoderAggregate2(nn.Module):
    def __init__(self, hidden_dim1, hidden_dim2, dropout):
        super(EncoderAggregate2, self).__init__()

        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2

        self.layer2 = nn.Linear(2 * hidden_dim1, hidden_dim2, bias=False)

        torch.nn.init.xavier_uniform_(self.layer2.weight)

        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, feature_emd, adj_layer2, feature_layer2):
        """
        :param layer1_nodes:
        :param feature_emd: h^{k-1}_{v}
        :param adj_layer2:
        :param feature_layer2: h^{k-1}_{Neighbor(v)}
        :return:
        """

        aggregated_0 = torch.matmul(adj_layer2, feature_layer2)

        aggregated_0 = aggregated_0 / 2

        retain_aggregated = aggregated_0

        aggregated_0 = aggregated_0.reshape(feature_emd.shape[0], -1, self.hidden_dim1).mean(
            dim=1)  # Of shape torch.size([|B|, hidden_dim_1])

        combined = torch.cat((feature_emd, aggregated_0), dim=1)

        embeddings = self.relu(self.layer2(combined))

        return embeddings, retain_aggregated

class EncoderAggregateOthers(nn.Module):
    def __init__(self, hidden_dim1, hidden_dim2, dropout):
        super(EncoderAggregateOthers, self).__init__()

        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2

        self.layer3 = nn.Linear(2 * hidden_dim1, hidden_dim2, bias=False)

        torch.nn.init.xavier_uniform_(self.layer3.weight)

        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, feature_emd, adj_layer2, feature_layer2, emb_list):
        """
        :param layer1_nodes:
        :param feature_emd: h^{k-1}_{v}
        :param adj_layer2:
        :param feature_layer2: h^{k-1}_{Neighbor(v)}
        :return:
        """

        aggregated_0 = torch.matmul(adj_layer2, feature_layer2)

        aggregated_0 = aggregated_0 / 2

        retain_aggregated = aggregated_0

        # simulate the message passing process
        for emb_mat in reversed(emb_list):
            aggregated_0 = aggregated_0.reshape(emb_mat.shape[0], -1, self.hidden_dim1).mean(
                dim=1)  # Of shape torch.size([|B|, hidden_dim_1])

        combined = torch.cat((feature_emd, aggregated_0), dim=1)

        embeddings = self.relu(self.layer3(combined))

        return embeddings, retain_aggregated

class AttentionLayer1(nn.Module):
    def __init__(self, feat_dim, attention_dim):
        super(AttentionLayer1, self).__init__()

        self.feat_dim = feat_dim
        self.hidden_dim = 64
        self.attention_dim = attention_dim
        self.attention_layer1 = nn.Linear(feat_dim, 64, bias=False)
        self.attention_layer2 = nn.Linear(64, attention_dim, bias=False)
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat_emd):
        attention_emd = self.attention_layer1(feat_emd)
        attention_emd = self.relu(attention_emd)
        attention_emd = self.attention_layer2(attention_emd)
        attention_emd = self.sigmoid(attention_emd)
        return attention_emd


class AttentionLayer2(nn.Module):
    def __init__(self, feat_dim, attention_dim):
        super(AttentionLayer2, self).__init__()

        self.hidden_dim = 64
        self.attention_dim = attention_dim
        self.attention_layer1 = nn.Linear(feat_dim, 64, bias=False)
        self.attention_layer2 = nn.Linear(64, attention_dim, bias=False)
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat_emd):
        attention_emd = self.attention_layer1(feat_emd)
        attention_emd = self.relu(attention_emd)
        attention_emd = self.attention_layer2(attention_emd)
        attention_emd = self.sigmoid(attention_emd)
        return attention_emd


class Gps(nn.Module):
    """
    Network that consolidates all into a single nn.Module
    """

    def __init__(self, feat_dim, sage_hidden_dim1, node_embedding_dim, sage_dropout,sim_func = "dot"):
        super(Gps, self).__init__()
        self.attention1 = AttentionLayer1(feat_dim=feat_dim, attention_dim=256)
        self.attention2 = AttentionLayer2(feat_dim=256, attention_dim=256)
        self.step1 = EncoderAggregate1(feat_dim, 256, node_embedding_dim, sage_dropout)
        self.step2 = EncoderAggregate2(256, node_embedding_dim, sage_dropout)
        self.step3 = EncoderAggregateOthers(256, node_embedding_dim, sage_dropout)
        self.sim_func = sim_func
        self.top_order = True
        self.lamda = 1

        if "_" in sim_func:
            config = sim_func.split("_")
            self.sim_func = config[0]
            self.lamda = float(config[1])

        if self.sim_func == "dist":
            self.top_order = False

    def forward(self, comp_adj, node_ids, feature_matrix, topn=2, depth=2):
        seed_nodes = node_ids.flatten()

        adj_layer1, feature_layer1, nodes_layer1 = self.policy_driven_traverse_first(comp_adj, seed_nodes, feature_matrix,
                                                                                     topn)
        node_embeddings = self.step1(seed_nodes, feature_matrix, adj_layer1, feature_layer1)

        adj_layer2, feature_layer2, nodes_layer_others = self.policy_driven_traverse_others(comp_adj, nodes_layer1, node_embeddings,
                                                                                      topn)
        node_embeddings, agg_embeddings = self.step2(node_embeddings, adj_layer2, feature_layer2)

        if depth > 2:
            emb_list = [node_embeddings, agg_embeddings]
            for dp in range(depth - 2):
                adj_layer2, feature_layer2, nodes_layer_others = self.policy_driven_traverse_others(comp_adj, nodes_layer_others,
                                                                                              node_embeddings,
                                                                                              topn)
                node_embeddings, embeddings = self.step3(node_embeddings, adj_layer2, feature_layer2, emb_list)
                emb_list.append(embeddings)

        return node_embeddings


    def l1_norm(self,x, y):
        """
        Input: x is a Nxd matrix
               y is a Mxd matirx
        Output: l1_norm is a NxM matrix where dist[i,j] is the l1 norm between x[i,:] and y[j,:]

        i.e. dist[i,j] = |x[i,:]-y[j,:]|
        """
        x_num = x.shape[0]
        y_num = y.shape[0]
        rlt = [torch.sum(torch.abs(x[i, :] - y[j, :])).item() for i in range(x_num) for j in range(y_num)]
        return torch.Tensor(rlt).reshape(x_num, y_num)

    def l2_norm(self, x, y=None):
        '''
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''
        x_num = x.shape[0]
        y_num = y.shape[0]
        rlt = [torch.norm(x[i, :] - y[j, :],2).item() for i in range(x_num) for j in range(y_num)]
        return torch.Tensor(rlt).reshape(x_num,y_num)

    def sim_matrix(self, a, b, eps=1e-8):
        """
        added eps for numerical stability
        """
        if self.sim_func == "cos":
            a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
            a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
            b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
            sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        elif self.sim_func == "dist":
            sim_mt_l2 = self.l2_norm(a, b)
            sim_mt_l1 = self.l1_norm(a, b)
            sim_mt = self.lamda * sim_mt_l2 + (1 - self.lamda) * sim_mt_l1

        elif self.sim_func == "kernel":
            gamma = a.shape[1]
            sim_mt_l2 = self.l2_norm(a, b)
            rbf_sim = torch.exp(-1/gamma*sim_mt_l2)
            sim_mt_l1 = self.l1_norm(a, b)
            laplacian_sim = torch.exp(-1/gamma*sim_mt_l1)
            sim_mt = self.lamda * rbf_sim + (1 - self.lamda) * laplacian_sim
        return sim_mt


    def policy_driven_traverse_first(self, compact_adj, seed_nodes, feature_emb, topn):

        node_neighs = self.dict_node_connections(compact_adj)

        adj1, feature_emb1, layer1_nodes = self.attention_based_sampling(seed_nodes, feature_emb,
                                                                         node_neighs, topn,
                                                                         0)
        return adj1, feature_emb1, layer1_nodes

    def policy_driven_traverse_others(self, compact_adj, nodes_layer1, feature_emb, topn):

        node_neighs = self.dict_node_connections(compact_adj)

        flatten_nodes_layer = []

        for each in nodes_layer1:
            for _ in each:
                flatten_nodes_layer.append(_)

        adj2, feature_emb2, layer2_nodes = self.attention_based_sampling(flatten_nodes_layer, feature_emb,
                                                                         node_neighs, topn,
                                                                         1)
        return adj2, feature_emb2, layer2_nodes

    def dict_node_connections(self, compact_adj):
        node_dict = {}
        full_adj = compact_adj.adj.todense().A
        for node_id, all_neighs in enumerate(full_adj):
            if node_id not in node_dict.keys():
                neighs_ids = set()
                for neigh_idx, connection in enumerate(all_neighs):
                    if connection == 1:
                        neighs_ids.add(neigh_idx)
                node_dict[node_id] = neighs_ids
        return node_dict

    def attention_based_sampling(self, pre_nodes, feature_emd, neighs, top_n, layer=0):

        if layer == 0:
            samp_neighs = [neighs[node] for node in pre_nodes]
            unique_nodes_list = list(set.union(*samp_neighs))
            unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
            adj_matrix = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
            column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
            row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
            adj_matrix[row_indices, column_indices] = 1
            neighs_emd = feature_emd[unique_nodes_list]
            neighs_trans = self.attention1(neighs_emd)
            if self.sim_func == "dot":
                alphas = torch.matmul(torch.abs(feature_emd[pre_nodes]), torch.transpose(neighs_trans, 0, 1))
            else:
                alphas = self.sim_matrix(torch.abs(feature_emd[pre_nodes]), neighs_trans)
            weighted_adj = adj_matrix * alphas
            mask = torch.zeros(weighted_adj.shape[0], weighted_adj.shape[1])
            if weighted_adj.shape[1] < top_n:
                kvals, kidx = weighted_adj.topk(k=weighted_adj.shape[1], dim=1, largest=self.top_order)
            else:
                kvals, kidx = weighted_adj.topk(k=top_n, dim=1, largest=self.top_order)
            mask[torch.arange(mask.size(0))[:, None], kidx] = 1
            weighted_adj = mask * weighted_adj
            layer1_nodes = []
            for row_idx, each_row in enumerate(weighted_adj):
                row_neighs = []
                for column_idx, each_value in enumerate(each_row):
                    if each_value != 0:
                        row_neighs.append(column_idx)
                if len(row_neighs) < top_n and len(row_neighs) > 0:
                    for _ in range(top_n - len(row_neighs)):
                        row_neighs.append(np.random.choice(row_neighs))
                if len(row_neighs) == 0:
                    for _ in range(top_n):
                        row_neighs.append(np.random.choice(list(range(weighted_adj.shape[1]))))
                layer1_nodes.append(row_neighs)
            return weighted_adj, neighs_emd, layer1_nodes

        else:
            samp_neighs = [neighs[node] for node in pre_nodes]
            unique_nodes_list = list(set.union(*samp_neighs))
            unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
            adj_matrix = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
            column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
            row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
            adj_matrix[row_indices, column_indices] = 1
            neighs_emd = feature_emd[unique_nodes_list]
            neighs_trans = self.attention2(neighs_emd)
            if self.sim_func == "dot":
                alphas = torch.matmul(torch.abs(feature_emd[pre_nodes]), torch.transpose(neighs_trans, 0, 1))
            else:
                alphas = self.sim_matrix(torch.abs(feature_emd[pre_nodes]), neighs_trans)
            weighted_adj = adj_matrix * alphas
            mask = torch.zeros(weighted_adj.shape[0], weighted_adj.shape[1])
            if weighted_adj.shape[1] < top_n:
                kvals, kidx = weighted_adj.topk(k=weighted_adj.shape[1], dim=1, largest=self.top_order)
            else:
                kvals, kidx = weighted_adj.topk(k=top_n, dim=1, largest=self.top_order)
            mask[torch.arange(mask.size(0))[:, None], kidx] = 1
            weighted_adj = mask * weighted_adj
            layer_nodes = []
            for row_idx, each_row in enumerate(weighted_adj):
                row_neighs = []
                for column_idx, each_value in enumerate(each_row):
                    if each_value != 0:
                        row_neighs.append(column_idx)
                if len(row_neighs) < top_n and len(row_neighs) > 0:
                    for _ in range(top_n - len(row_neighs)):
                        row_neighs.append(np.random.choice(row_neighs))
                if len(row_neighs) == 0:
                    for _ in range(top_n):
                        row_neighs.append(np.random.choice(list(range(weighted_adj.shape[1]))))
                layer_nodes.append(row_neighs)
            return weighted_adj, neighs_emd, layer_nodes
