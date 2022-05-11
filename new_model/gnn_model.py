import torch
import torch.nn as nn
import torch.nn.functional as F
from model.gps_model import Gps
from Utlis.inter_intra_utlis import CompactAdjacency
from scipy.sparse import csr_matrix
import numpy as np


class GCN(nn.Module):
    """
    Graph Convolutional Network based on https://arxiv.org/abs/1609.02907

    """

    def __init__(self, feat_dim, hidden_dim1, hidden_dim2, dropout, is_sparse=False):
        """Dense version of GAT."""
        super(GCN, self).__init__()
        # self.dropout = dropout
        self.W1 = nn.Parameter(torch.FloatTensor(feat_dim, hidden_dim1))
        self.W2 = nn.Parameter(torch.FloatTensor(hidden_dim1, hidden_dim2))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

        nn.init.xavier_uniform_(self.W1.data)
        nn.init.xavier_uniform_(self.W2.data)

        self.is_sparse = is_sparse

    def forward(self, x, adj):
        # Layer 1
        support = torch.mm(x, self.W1)
        embeddings = torch.sparse.mm(adj, support) if self.is_sparse else torch.mm(adj, support)
        embeddings = self.relu(embeddings)
        embeddings = self.dropout(embeddings)

        # Layer 2
        support = torch.mm(embeddings, self.W2)
        embeddings = torch.sparse.mm(adj, support) if self.is_sparse else torch.mm(adj, support)
        embeddings = self.relu(embeddings)

        return embeddings


class GraphSage(nn.Module):
    """
    GraphSAGE model (https://arxiv.org/abs/1706.02216) to learn the role of atoms in the molecules inductively.
    Transforms input features into a fixed length embedding in a vector space. The embedding captures the role.
    """

    def __init__(self, feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GraphSage, self).__init__()

        self.feat_dim = feat_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2

        self.layer1 = nn.Linear(2 * feat_dim, hidden_dim1, bias=False)
        self.layer2 = nn.Linear(2 * hidden_dim1, hidden_dim2, bias=False)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, forest, feature_matrix):
        feat_0 = feature_matrix[forest[0]]  # Of shape torch.Size([|B|, feat_dim])
        feat_1 = feature_matrix[forest[1]]  # Of shape torch.size(|B|, fanouts[0], feat_dim)

        # Depth 1
        x = feature_matrix[forest[1]].mean(dim=1)  # Of shape torch.size(|B|, feat_dim)
        feat_0 = torch.cat((feat_0, x), dim=1)  # Of shape torch.size(|B|, 2 * feat_dim)
        feat_0 = self.relu(self.layer1(feat_0))  # Of shape torch.size(|B|, hidden_dim1)
        feat_0 = self.dropout(feat_0)

        # Depth 2
        x = feature_matrix[forest[2]].mean(dim=1)  # Of shape torch.size(|B|*fanouts[0], feat_dim)
        feat_1 = torch.cat((feat_1.reshape(-1, self.feat_dim), x),
                           dim=1)  # Of shape torch.size(|B|*fanouts[0], 2 * feat_dim)
        feat_1 = self.relu(self.layer1(feat_1))  # Of shape torch.size(|B|*fanouts[0], hidden_dim1)
        feat_1 = self.dropout(feat_1)

        # Combine
        feat_1 = feat_1.reshape(forest[0].shape[0], -1, self.hidden_dim1).mean(
            dim=1)  # Of shape torch.size([|B|, hidden_dim_1])
        combined = torch.cat((feat_0, feat_1), dim=1)  # Of shape torch.Size(|B|, 2 * hidden_dim1)
        embeddings = self.relu(self.layer2(combined))  # Of shape torch.Size(|B|, hidden_dim2)

        return embeddings


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, feat_dim, hidden_dim1, hidden_dim2, dropout, alpha = 0.2, nheads = 2):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = nn.ModuleList(
            [GraphAttentionLayer(feat_dim, hidden_dim1, dropout=dropout, alpha=alpha, concat=True) for _ in
             range(nheads)])
        self.out_att = GraphAttentionLayer(hidden_dim1 * nheads, hidden_dim2, dropout=dropout, alpha=alpha,
                                           concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        embeddings = F.elu(self.out_att(x, adj))  # Node embeddings

        return embeddings


class GraphEmb(nn.Module):
    """
    This module learns a single graph level representation for a molecule given GNN generated node embeddings
    """

    def __init__(self, attr_dim, embedding_dim, hidden_dim, output_dim):
        super(GraphEmb, self).__init__()
        self.attr_dim = attr_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.layer1 = nn.Linear(attr_dim + embedding_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

        nn.init.xavier_normal_(self.layer1.weight)
        nn.init.xavier_normal_(self.layer2.weight)

    def forward(self, node_features, node_embeddings, mode="mean"):
        """
        :param node_features:  initial node feature
        :param node_embeddings:  node embedding after message passing
        :param mode: how to get graph representation
        :return: graph embedding
        """
        combined_rep = torch.cat((node_features, node_embeddings),
                                 dim=1)  # Concat initial node attributed with embeddings from sage
        hidden_rep = self.act(self.layer1(combined_rep))
        hidden_rep_1 = self.act(self.layer2(hidden_rep))  # Generate final graph level embedding
        if mode == "mean":
            graph_rep = torch.mean(hidden_rep_1, dim=0)
        return graph_rep


class GraphBlock(nn.Module):
    """
    Graph Block contains gcn/gat/graphsage and get node representation and graph representation after message passing
    """

    def __init__(self, graph_mode, feat_dim=16, hidden_dim=16, node_embedding_dim=16, dropout=0.1,
                 readout_hidden_dim=16, graph_embedding_dim=16, sparse_adj=False,gat_alpha = 0.2,gat_nheads = 2):
        super(GraphBlock, self).__init__()
        self.mode = graph_mode
        if graph_mode == "gcn":
            self.gnn = GCN(feat_dim, hidden_dim, node_embedding_dim, dropout, is_sparse=sparse_adj)
        elif graph_mode == "gat":
            self.gnn = GAT(feat_dim, hidden_dim, node_embedding_dim, dropout, alpha = gat_alpha, nheads = gat_nheads)
        elif graph_mode == "sage":
            self.gnn = GraphSage(feat_dim, hidden_dim, node_embedding_dim, dropout)
        elif graph_mode == "gps":
            self.gnn = Gps(feat_dim,hidden_dim,node_embedding_dim,dropout)
        else:
            assert 0,"not support gnn mode"
        self.graph_learning = GraphEmb(feat_dim, node_embedding_dim, readout_hidden_dim, graph_embedding_dim)

    def forward(self, adj_matrix, feature_matrix,supply):
        if self.mode == "sage":
            node_ids,forest = supply
            node_emb = self.gnn(forest,feature_matrix)
            graph_emb = self.graph_learning(feature_matrix, node_emb)
        elif self.mode == "gps":
            node_ids, forest = supply
            adj_matrix = np.ones([feature_matrix.shape[0],feature_matrix.shape[0]])
            comp_adj = CompactAdjacency(csr_matrix(adj_matrix))
            node_emb = self.gnn(comp_adj,node_ids,feature_matrix)
            graph_emb = self.graph_learning(feature_matrix, node_emb)
        else:
            node_emb = self.gnn(feature_matrix, adj_matrix)
            graph_emb = self.graph_learning(feature_matrix, node_emb)
        return node_emb, graph_emb