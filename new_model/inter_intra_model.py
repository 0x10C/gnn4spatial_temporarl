import torch
import torch.nn as nn
import torch.nn.functional as F
from model.gnn_model import GraphBlock

class IntraGraphBlock(nn.Module):
    """
    GNN graph learning in one timestamp
    apt_cor  --> W_{intra}

    """

    def __init__(self, num_node_emb, graph_mode, feat_dim, hidden_dim, node_embedding_dim \
                 , dropout, readout_hidden_dim, graph_embedding_dim):
        super(IntraGraphBlock, self).__init__()
        self.mode = graph_mode
        self.num_node_emb = num_node_emb
        # self.apt_cor = nn.Parameter(torch.FloatTensor(num_node_emb, num_node_emb))
        # nn.init.xavier_normal_(self.apt_cor.data)
        self.atten_layer1 = nn.Linear(num_node_emb,num_node_emb)
        self.atten_layer2 = nn.Linear(num_node_emb,num_node_emb)
        nn.init.xavier_normal_(self.atten_layer1.weight)
        nn.init.xavier_normal_(self.atten_layer1.weight)

        self.gnn = GraphBlock(graph_mode, feat_dim, hidden_dim, node_embedding_dim \
                              , dropout, readout_hidden_dim, graph_embedding_dim)

    def forward(self, init_node_emb, supply):
        # adj = torch.mm(torch.matmul(init_node_emb, self.apt_cor), init_node_emb.T) \
        #       / math.sqrt(self.num_node_emb)
        adj = torch.mm(self.atten_layer1(init_node_emb),self.atten_layer2(init_node_emb).T)
        adj_norm = F.softmax(adj, dim=1)
        node_emb, graph_emb = self.gnn(adj_norm, init_node_emb,supply)
        return node_emb, graph_emb


class MixtureGraphEmbedding(nn.Module):
    """
    Mixture node embedding with graph embedding. Concat or Linear Transformation ...

    """

    def __init__(self, mode, node_emb):
        super(MixtureGraphEmbedding, self).__init__()
        self.mode = mode
        self.layer = nn.Linear(node_emb * 2, node_emb)
        nn.init.xavier_normal_(self.layer.weight)
        self.drop = nn.Dropout(p = 0.1)
        self.linear = nn.Linear(node_emb,node_emb)
        self.norm = nn.LayerNorm([10,node_emb])

    def forward(self, node_emb, graph_emb):
        """

        :param node_emb: #node * #emb
        :param graph_emb: 1 * #emb
        :return:
        """
        if self.mode == "cat":
            node_num = node_emb.shape[-2]
            graph_emb_align = graph_emb.repeat(node_num, 1)

            mixture_node_emb = torch.cat((node_emb, graph_emb_align), dim=1)
            output = self.norm(self.layer(mixture_node_emb))
            # output = self.layer(mixture_node_emb)

        return output


class InterGraphBlock(nn.Module):
    """
        GNN graph learning between two timestamps

    """

    def __init__(self, num_node_emb, graph_mode, feat_dim, hidden_dim, node_embedding_dim \
                 , dropout, readout_hidden_dim, graph_embedding_dim):
        super(InterGraphBlock, self).__init__()
        self.mode = graph_mode
        self.num_node_emb = num_node_emb
        # self.apt_cor = nn.Parameter(torch.FloatTensor(num_node_emb, num_node_emb))
        # nn.init.xavier_uniform_(self.apt_cor.data)
        self.atten_layer1 = nn.Linear(num_node_emb,num_node_emb)
        self.atten_layer2 = nn.Linear(num_node_emb,num_node_emb)
        nn.init.xavier_normal_(self.atten_layer1.weight)
        nn.init.xavier_normal_(self.atten_layer1.weight)

        self.gnn = GraphBlock(graph_mode, feat_dim, hidden_dim, node_embedding_dim \
                              , dropout, readout_hidden_dim, graph_embedding_dim)

    def forward(self, pre_node_emb, cur_node_emb):
        # adj = torch.mm(torch.mm(pre_node_emb, self.apt_cor), cur_node_emb.T) \
        #         #       / math.sqrt(self.num_node_emb)
        adj = torch.mm(self.atten_layer1(pre_node_emb),self.atten_layer2(cur_node_emb).T)

        adj_norm = F.softmax(adj, dim=1)
        node_emb, _ = self.gnn(adj_norm, pre_node_emb,supply = None)
        return node_emb


class MixtureOutput(nn.Module):
    """
    Mixture inter embedding and intra embedding . Concat or Linear Transformation ...

    """

    def __init__(self, node_emb, hidden_emb, output_emb, mode="cat"):
        super(MixtureOutput, self).__init__()
        self.mode = mode
        self.layer1 = nn.Linear(node_emb * 2, hidden_emb)
        self.layer2 = nn.Linear(hidden_emb, hidden_emb)
        self.output = nn.Linear(hidden_emb, output_emb)
        self.act = nn.LeakyReLU()
        self.drop = nn.Dropout(p = 0.1)
        nn.init.xavier_normal_(self.layer1.weight)
        nn.init.xavier_normal_(self.layer2.weight)
        nn.init.xavier_normal_(self.output.weight)

    def forward(self, node_emb_inter, node_emb_intra):
        """

        :param node_emb_inter: #node * #inter_emb
        :param node_emb_intra: #node * #intra_emb
        :return:
        """
        if self.mode == "cat":
            mixture_emb = torch.cat((node_emb_inter, node_emb_intra), dim=1)
        hidden_emb = self.act(self.drop(self.layer1(mixture_emb)))
        # hidden_emb = self.act(self.layer2(hidden_emb))
        return hidden_emb


class TimeBlock(nn.Module):
    """
        one timeblock need two consecutive timestamp data<pre_time,cur_time>
        return cur time node embedding after inter and intra gnn block

    """

    def __init__(self, pre_node_fea, intra_pre_gnn_mode, cur_node_fea, intra_cur_gnn_mode, \
                 pre_mix_mode, cur_mix_mode, inter_node_fea, inter_gnn_mode, out_node_emb, \
                 out_hidden, out_rlt_mode, intra_gnn_fea, intra_gnn_hidden, intra_gnn_node_emb, \
                 intra_gnn_drop, intra_gnn_readout, intra_gnn_graph_emb, inter_gnn_fea, inter_gnn_hidden,
                 inter_gnn_node_emb, \
                 inter_gnn_drop, inter_gnn_readout, inter_gnn_graph_emb):
        super(TimeBlock, self).__init__()

        self.intra_layer_pre = IntraGraphBlock(pre_node_fea, intra_pre_gnn_mode, intra_gnn_fea, intra_gnn_hidden,
                                               intra_gnn_node_emb, \
                                               intra_gnn_drop, intra_gnn_readout, intra_gnn_graph_emb)

        self.intra_layer_cur = IntraGraphBlock(cur_node_fea, intra_cur_gnn_mode, intra_gnn_fea, intra_gnn_hidden,
                                               intra_gnn_node_emb, \
                                               intra_gnn_drop, intra_gnn_readout, intra_gnn_graph_emb)

        self.pre_mix_graph = MixtureGraphEmbedding(pre_mix_mode, 256)
        self.cur_mix_graph = MixtureGraphEmbedding(cur_mix_mode, 256)
        self.inter_layer = InterGraphBlock(256, inter_gnn_mode, 256, inter_gnn_hidden,
                                           inter_gnn_node_emb, \
                                           inter_gnn_drop, inter_gnn_readout, inter_gnn_graph_emb)
        self.output = MixtureOutput(out_node_emb, out_hidden, out_rlt_mode)

    def forward(self, pre_node, cur_node,pre_node_supply,cur_node_supply):
        pre_node_emb, pre_graph_emb = self.intra_layer_pre(pre_node,pre_node_supply)
        cur_node_emb, cur_graph_emb = self.intra_layer_cur(cur_node,cur_node_supply)
        pre_mix_emb = self.pre_mix_graph(pre_node_emb, pre_graph_emb)
        cur_mix_emb = self.cur_mix_graph(cur_node_emb, cur_graph_emb)
        node_inter_emb = self.inter_layer(pre_mix_emb, cur_mix_emb)
        node_output_emb = self.output(node_inter_emb, cur_node_emb)
        return node_output_emb


class Model(nn.Module):
    """
    multi-timestamp training
    todo:get_laplacien
    """

    def __init__(self, win_size, pre_node_fea, intra_pre_gnn_mode, cur_node_fea, intra_cur_gnn_mode, \
                 pre_mix_mode, cur_mix_mode, inter_node_fea, inter_gnn_mode, out_node_emb, \
                 out_hidden, out_rlt_mode, intra_gnn_fea, intra_gnn_hidden, intra_gnn_node_emb, \
                 intra_gnn_drop, intra_gnn_readout, intra_gnn_graph_emb, inter_gnn_fea, inter_gnn_hidden,
                 inter_gnn_node_emb, \
                 inter_gnn_drop, inter_gnn_readout, inter_gnn_graph_emb,final_emb,num_cats):
        super(Model, self).__init__()
        self.win_size = win_size
        self.output = nn.Linear(final_emb, num_cats)
        self.act = nn.ReLU()
        # self.drop = nn.Dropout(p = 0.1)
        nn.init.xavier_normal_(self.output.weight)
        # self.stock_block = nn.ModuleList()

        # todo: same timeblock in every timestamp
        # self.stock_block.extend([TimeBlock(pre_node_fea, intra_pre_gnn_mode, cur_node_fea, intra_cur_gnn_mode, \
        #                                    pre_mix_mode, cur_mix_mode, inter_node_fea, inter_gnn_mode, out_node_emb, \
        #                                    out_hidden, out_rlt_mode, intra_gnn_fea, intra_gnn_hidden,
        #                                    intra_gnn_node_emb, \
        #                                    intra_gnn_drop, intra_gnn_readout, intra_gnn_graph_emb, inter_gnn_fea,
        #                                    inter_gnn_hidden, inter_gnn_node_emb, \
        #                                    inter_gnn_drop, inter_gnn_readout, inter_gnn_graph_emb) for i in
        #                          range(win_size)])

        self.time_cursor = TimeBlock(pre_node_fea, intra_pre_gnn_mode, cur_node_fea, intra_cur_gnn_mode, \
                                     pre_mix_mode, cur_mix_mode, inter_node_fea, inter_gnn_mode, out_node_emb, \
                                     out_hidden, out_rlt_mode, intra_gnn_fea, intra_gnn_hidden,
                                     intra_gnn_node_emb, \
                                     intra_gnn_drop, intra_gnn_readout, intra_gnn_graph_emb, inter_gnn_fea,
                                     inter_gnn_hidden, inter_gnn_node_emb, \
                                     inter_gnn_drop, inter_gnn_readout, inter_gnn_graph_emb)

    def forward(self, node_fea,supply,task = "regression"):
        for i in range(self.win_size):
            if i == 0:
                pre_node = node_fea[0]
                cur_node = node_fea[1]
                pre_node_supply = supply[0]
                cur_node_supply = supply[1]
                cur_node_emb = self.time_cursor(pre_node, cur_node,pre_node_supply,cur_node_supply)
            else:

                cur_node = node_fea[i + 1].contiguous()
                pre_node_supply = cur_node_supply
                cur_node_supply = supply[i+1]
                pre_node = cur_node_emb.contiguous()
                cur_node_emb = self.time_cursor(pre_node, cur_node,pre_node_supply,cur_node_supply)
        logits = torch.mean(self.output(cur_node_emb), dim=0)
        # if task == "regression":
        #     logits = self.output(cur_node_emb)

        return logits


if __name__ == "__main__":
    test = torch.randn(9, 10, 16)
    print(test)

    gold = torch.randn(16)
    args = {"win_size": 2,
            "pre_node_fea": 16,
            "intra_pre_gnn_mode": "gcn",
            "cur_node_fea": 16,
            "intra_cur_gnn_mode": "gcn",
            "pre_mix_mode": "cat",
            "cur_mix_mode": "cat",
            "inter_node_fea": 16,
            "inter_gnn_mode": "gcn",
            "out_node_emb": 16,
            "out_hidden": 16,
            "out_rlt_mode": 16,
            "intra_gnn_fea": 16,
            "intra_gnn_hidden": 16,
            "intra_gnn_node_emb": 16,
            "intra_gnn_drop": 0.5,
            "intra_gnn_readout": 16,
            "intra_gnn_graph_emb": 16,
            "inter_gnn_fea": 16,
            "inter_gnn_hidden": 16,
            "inter_gnn_node_emb": 16,
            "inter_gnn_drop": 0.5,
            "inter_gnn_readout": 16,
            "inter_gnn_graph_emb": 16,
            "final_emb":256,
            "num_cats":5
            }
    model = Model(**args)
    rlt = model(test)
    criterion = torch.nn.L1Loss(reduction='mean')
    opt = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    opt.zero_grad()
    loss = criterion(rlt, gold)
    loss.backward()
    opt.step()
    print(rlt)
