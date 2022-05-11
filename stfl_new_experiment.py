import torch.utils.data
from torch import optim
from torch.nn import functional as F
from utlis.vae_utlis import get_data_feature,loss_function_generation
from utlis.inter_intra_utlis import *
import pandas as pd
import argparse
from utlis.vae_utlis import run_vae
from new_model.VAE import Generator_global_distribution,VAE
# from new_model.inter_intra_model import Model
from DataTransformer import Adjaency_Generator
from new_model.GAT import GatClassification
from new_model.GCN import GCNClassification
from new_model.Graphsage import GraphSageClassification
from new_model.GPS import Gps
import json



def get_raw_data(path):

    middle_data = np.load(path)

    train_feature = middle_data["train_feature"]
    val_feature = middle_data["val_feature"]
    train_targets = middle_data["train_targets"]
    val_targets = middle_data["val_targets"]
    return train_feature,val_feature,train_targets,val_targets


def split_data(train_feature,train_targets,val_feature,val_targets,n_client = 5):
    labels = train_targets.argmax(axis=1)
    labels_valid = val_targets.argmax(axis=1)
    clients_data = []
    clients_valid = []
    for i in range(n_client):
        clients_data.append(np.concatenate((
            train_feature[labels[labels==(i)%n_client]],
            train_feature[labels[labels==(i+1)%n_client]],
            train_feature[labels[labels==(i+2)%n_client]]
        )))
        clients_valid.append(np.concatenate((
            val_feature[labels_valid[labels_valid==(i)%n_client]],
            val_feature[labels_valid[labels_valid==(i+1)%n_client]],
            val_feature[labels_valid[labels_valid==(i+2)%n_client]]
        )))
    return clients_data,clients_valid

# todo:scaler use sklearning
def max_scale(clients_data,clients_valid):
    max_features = []
    for i, data in enumerate(clients_data):
        max_features.append(data.max())
        clients_data[i] = data / data.max()
        clients_valid[i] = clients_valid[i] / data.max()
        clients_valid[i][clients_valid[i] > 1] = 1
    return clients_data,clients_valid


def flatten_data(train_feature,val_feature):
    train_feature = train_feature.reshape((train_feature.shape[0], train_feature.shape[1] * train_feature.shape[2]))
    val_feature = val_feature.reshape((val_feature.shape[0], val_feature.shape[1] * val_feature.shape[2]))
    return train_feature,val_feature

def fed_vae_data_prepare(train_feature, val_feature, train_targets, val_targets,n_clients):
    train_feature, val_feature = flatten_data(train_feature, val_feature)
    clients_data, clients_valid = split_data(train_feature, train_targets, val_feature, val_targets, n_clients)
    clients_data, clients_valid = max_scale(clients_data, clients_valid)
    print("FL-vae data ready")
    return clients_data, clients_valid


def get_distribution_repr(clients_data):
    local_features = []
    for i in range(5):
        model_path = "log/vae/Model/VAE_Client{}.pth.tar".format(i)
        dataloader = torch.utils.data.DataLoader(clients_data[i], batch_size=64, shuffle=True)
        local_features.append(get_data_feature(model_path, dataloader))
    return local_features

def gusses_model_initialization(model,n_clients,device = "cpu"):
    gusses_models = []
    opts = []
    for i in range(n_clients):
        gusses_model = model().to(device)
        optimizer = optim.Adam(gusses_model.parameters(), lr=5e-4)
        gusses_models.append(gusses_model)
        opts.append(optimizer)
    for gusses_model in gusses_models:
        gusses_model.train()
    print("FL-guesses model Initialized!")
    return gusses_models,opts


def loss_function(recon_x, x):
        # BCE = F.binary_cross_entropy(global_distribution, gusses[i], reduction='sum')
        MSE = torch.sum((recon_x - x).pow(2))
        return MSE


def KLD(recon_x, x):
        # return -torch.sum(recon_x*(recon_x.log()-x.log()))
        return F.kl_div(recon_x.log(), x, reduction='sum')


def cosine_distance(a, b):
        return 1-(a * b).sum() / ((a ** 2).sum().pow(0.5) * (b ** 2).sum().pow(0.5))


def attentions(global_feature, local_features, distance_function):
        distance = []
        for local_feature in local_features:
            distance.append(distance_function(global_feature, local_feature))
        distance = torch.tensor(distance)
        return nn.Softmax(dim=0)(distance)

def fl_gnn_split_data(feature,label,supply,adj,n_clients):
    labels = label.argmax(axis=1)
    indexs = np.arange(len(labels))
    clients_data = []
    for i in range(n_clients):
        mask = np.concatenate((
            indexs[labels == (i) % n_clients],
            indexs[labels == (i + 1) % n_clients],
            indexs[labels == (i + 2) % n_clients]
        ))
        # print(mask)
        clients_data.append((feature[mask], label[mask], supply[mask],adj[mask]))

    for i in range(n_clients):
        print(clients_data[i][1].sum(axis=0))
    return clients_data

def get_adj(case_name,path,feature):
    adj_generator = Adjaency_Generator(case_name)

    if not os.path.exists(path['save']):
        os.makedirs("result")
    case_path = path['save'] + case_name + "/"
    if not os.path.exists(case_path):
        os.makedirs(case_path)
    path['save'] = case_path
    # np.save(path['save'] + "labels.npy", label)
    # output = open(path['save'] + "feature_matrices.pkl", "wb")
    # pickle.dump(feature, output)

    adj_generator.get_adj(len(feature), feature, path)




def get_gnn_raw_data(path,n_clients,case_name,split_ratio = 0.8):
    data_process = DataProcess(path)
    feature, label = data_process.read_data()

    check_path = path['save'] + case_name + "/" + "adjacency_matrices.pkl"
    if not os.path.exists(check_path):
        get_adj(case_name,path,feature)

    with open(check_path, 'rb') as f:
        adj_matrices = pickle.load(f)

    adj_matrices = np.array([ele.A for ele in adj_matrices])

    n_sample = len(feature)
    train_index = int(n_sample*split_ratio)
    feature = np.array(feature)
    feature_train = feature[:train_index]
    label_train = label[:train_index]
    feature_test  = feature[train_index:]
    label_test = label[train_index:]

    adj_matrices_train = adj_matrices[:train_index]
    adj_matrices_test = adj_matrices[train_index:]

    supplement_sage_gps_train = adpator_sage_gps(feature_train)
    supplement_sage_gps_test = adpator_sage_gps(feature_test)

    supplement_sage_gps_train =  np.array(supplement_sage_gps_train)
    clients_data = fl_gnn_split_data(feature_train,label_train,supplement_sage_gps_train,adj_matrices_train,n_clients)

    test_data = (feature_test,label_test,supplement_sage_gps_test,adj_matrices_test)
    print("FL-gnn data ready")
    return clients_data,test_data

# def test_data_process(test_data_clients, test_label_clients,test_supplement_clients):
#     rlt_data = []
#     rlt_label = []
#     rlt_supply = []
#     n_clients = len(test_data_clients)
#     len_list = [test_data_clients[i].shape[0] for i in range(n_clients)]
#     for i in range(n_clients):
#         rlt_data.append(test_data_clients[i])
#         rlt_label.append(test_label_clients[i])
#         rlt_supply.append(test_supplement_clients[i])
#     rlt_data = np.concatenate(rlt_data,0)
#     rlt_label = np.concatenate(rlt_label,0)
#     rlt_supply = np.concatenate(rlt_supply,0)
#
#     shuffle = [i for i in range(rlt_data.shape[0])]
#     random.shuffle(shuffle)
#     len_span_list = [0]
#     for i in range(len(len_list)):
#         len_span_list.append(len_list[i]+len_span_list[i])
#
#     test_data_clients, test_label_clients, test_supplement_clients = [],[],[]
#     for i in range(n_clients):
#         test_data_clients.append(rlt_data[shuffle[len_span_list[i]:len_span_list[i+1]]])
#         test_label_clients.append(rlt_label[shuffle[len_span_list[i]:len_span_list[i+1]]])
#         test_supplement_clients.append(rlt_supply[shuffle[len_span_list[i]:len_span_list[i+1]]])
#     return test_data_clients, test_label_clients,test_supplement_clients,rlt_data,rlt_label,rlt_supply

def get_model(args, feat_dim = 256, num_cats = 5):
    if args.model == 'gcn':
        model = GCNClassification(feat_dim, args.hidden_size, args.node_embedding_dim, args.dropout,
                                      args.readout_hidden_dim, args.graph_embedding_dim, num_cats,
                                      sparse_adj=args.sparse_adjacency)
    elif args.model == 'gat':
        model = GatClassification(feat_dim, args.hidden_size, args.node_embedding_dim, args.dropout,
                                      args.alpha, args.num_heads, args.readout_hidden_dim, args.graph_embedding_dim,
                                      num_cats)
    elif args.model == 'sage':
        model = GraphSageClassification(feat_dim, args.hidden_size, args.node_embedding_dim, args.dropout,
                                                  args.readout_hidden_dim, args.graph_embedding_dim, num_cats)
    elif args.model == "gps":
        model = Gps(feat_dim, args.hidden_size, args.node_embedding_dim, args.dropout,
                                  args.readout_hidden_dim, args.graph_embedding_dim, num_cats)

    else:
        raise Exception('No such model')

    return model


def fl_model_data_initialization(path,args,n_clients,case_name = "knn"):
    # clients_data partially non-iid
    clients_data,test_data = get_gnn_raw_data(path,n_clients,case_name,0.8)
    featrue_align_clients, label_clients, model_clients, opt_clients,adj_clients = [], [], [], [],[]
    supplement_clients = []
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    n_clients = len(clients_data)
    for i in range(n_clients):
        feature, label, supply,adj = clients_data[i]
        # supplement_sage_gps = adpator_sage_gps(feature)
        featrue_align_clients.append(feature)
        label_clients.append(label)
        supplement_clients.append(supply)
        adj_clients.append(adj)
        # train_index = int(len(featrue_align) * 0.75)
        # shuffle = [i for i in range(len(featrue_align))]
        # random.shuffle(shuffle)
        # test_data = featrue_align[shuffle[train_index:]]
        # test_label = label[shuffle[train_index:]]
        # test_supplement = supplement_sage_gps_align[shuffle[train_index:]]
        #
        # train_index_clients.append(train_index)
        # shuffle_clients.append(shuffle)
        # test_data_clients.append(test_data)
        # test_label_clients.append(test_label)
        # test_supplement_clients.append(test_supplement)
        model = get_model(args,256,5)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        model_clients.append(model)
        opt_clients.append(opt)
    # test_data_clients, test_label_clients, test_supplement_clients,test_global_data,test_global_label,test_global_supply \
    #     = test_data_process(test_data_clients,test_label_clients,test_supplement_clients)
    return featrue_align_clients, label_clients, model_clients, opt_clients,supplement_clients,criterion,test_data,adj_clients

def weight_aggregate(weight, global_model, participants):
    """
    This function has aggregation method 'mean'
    将全部子节点mobilenet模型的更新，同步到全局模型。
    上行和下行被部署在这个函数里面。
    更新方式为全部子节点权重取平均。
    Prune也在这部分进行。
    Prune：剪枝，把符合条件的权重置0

    Parameters:
    global_model    - 全局模型
    participants    - 全部子节点mobilenet模型

    Returns:
        无返回
    """

    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([participants[i].state_dict()[k].float()*weight[i] for i in range(len(participants))],0).sum(0)
    global_model.load_state_dict(global_dict)

    for model in participants:
        model.load_state_dict(global_model.state_dict())

def avg_aggregate(global_model,participants):

    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([participants[i].state_dict()[k].float() for i in range(len(participants))],0).mean(0)
    global_model.load_state_dict(global_dict)

    for model in participants:
        model.load_state_dict(global_model.state_dict())

def add_cmp_exper(Model,args_model,args,n_clients):
    single_model = Model(**args_model)
    global_model_avg = Model(**args_model)
    model_clients = []
    opt_clients = []
    for i in range(n_clients):
        model = Model(**args_model)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        model_clients.append(model)
        opt_clients.append(opt)
    opt_single = torch.optim.Adam(single_model.parameters(), lr=args.lr)
    return single_model,opt_single,model_clients,opt_clients,global_model_avg


def shuffle_train(feature,label,supply):
    n = len(feature)
    shuffle = [i for i in range(n)]
    random.shuffle(shuffle)
    return feature[shuffle],label[shuffle],supply[shuffle]

def run_fl_exp(args,path,Generator_global_distribution,run_vae,cmp_exp = False):
        device = args.device
        n_clients = args.num_clients
        case_name = args.case_name

        # log initialization
        train_loss_client_log = []
        test_metric_client_log = []
        generator_loss_client_log = []
        test_avg_log = []
        test_att_log = []
        test_single_log = []

        atten_log = []


        # get raw data
        train_feature, val_feature, train_targets, val_targets = get_raw_data(path['feature'])

        # data preprocess for vae
        clients_data_vae, clients_valid_vae = fed_vae_data_prepare(train_feature, val_feature, train_targets,
                                                                   val_targets, n_clients)

        # preprocess for gnn
        feature_align_clients, label_clients, model_clients, opt_clients, supplement_clients, criterion, test_data,adj_clients\
            = fl_model_data_initialization(path, args, n_clients,case_name)

        # initialization of vae
        if not os.path.exists("log/VAE/Model/VAE_Client1.pth.tar"):
            run_vae(clients_data_vae, clients_valid_vae)

        local_features = get_distribution_repr(clients_data_vae)
        # print(local_repr)

        # initialization of gusses model
        gusses_model, gusses_opts = gusses_model_initialization(Generator_global_distribution, n_clients)

        # initialization of global model
        global_model = get_model(args)

        # add compare experiment
        # if cmp_exp:
        #     single_model,single_opt,fl_avg_model_clients,fl_avg_opt_clients,global_model_avg = \
        #         add_cmp_exper(Model,args_model,args,n_clients)

        # trainning process

        test_f1_best = -1
        # print("win size = {}".format(args_model['win_size']))

        num_gnn_bp = args.num_gnn_bp

        train_len_clients = [len(ele) for ele in feature_align_clients]
        test_feature,test_label,test_supply,test_adj = test_data
        for i in range(args.epochs):
            print("epoch:{}".format(i))
            # prepare client data
            train_feature_clients = []
            train_label_clients = []
            train_supply_clients = []
            train_adj_clients = []
            for j in range(n_clients):
                # shuffle = shuffle_clients[j]
                # train_index = train_index_clients[j]
                train_label = label_clients[j]
                train_feature = feature_align_clients[j]
                train_supply = supplement_clients[j]
                train_adj = adj_clients[j]

                # train_feature,train_label,train_supply = shuffle_train(train_feature,train_label,train_supply)

                train_label = iter(train_label)
                train_feature = iter(train_feature)
                train_supply = iter(train_supply)
                train_adj = iter(train_adj)

                train_feature_clients.append(train_feature)
                train_label_clients.append(train_label)
                train_supply_clients.append(train_supply)
                train_adj_clients.append(train_adj)

            min_train_index = min(train_len_clients)
            print("min_train_index:{}".format(min_train_index))
            # calculate weight for one batch
            for b_i in range(int(min_train_index / args.batch_size)):
                gusses = []
                global_distribution = torch.zeros_like(local_features[0])
                for _ in range(int(args.batch_size/num_gnn_bp)):
                    for client_index in range(n_clients):
                            train_feature = train_feature_clients[client_index]
                            train_label = train_label_clients[client_index]
                            train_supply = train_supply_clients[client_index]
                            train_adj = train_adj_clients[client_index]

                            model = model_clients[client_index]
                            opt = opt_clients[client_index]
                            optimizer = opt
                            optimizer.zero_grad()
                            batch_loss = calculate_loss(model, train_feature, train_label, train_supply,train_adj, args.num_gnn_bp,
                                                        criterion,args.model)
                            batch_loss.backward()
                            optimizer.step()
                            batch_num = 256*b_i+8*_+i*min_train_index
                            # print("[gnn-stat]epoch:{},batch:{},client:{},loss:{}".format(i,batch_num, client_index,batch_loss))
                            temp_train_log = [i,batch_num,client_index,float(batch_loss.detach().numpy())]
                            train_loss_client_log.append(temp_train_log)


                for client_index in range(n_clients):
                    # gusses global distribution
                    local_distribution = local_features[client_index]
                    local_distribution.to(device)

                    gusess_distribution = gusses_model[client_index](local_distribution)
                    gusses.append(gusess_distribution)
                    global_distribution += gusess_distribution.data

                global_distribution /= n_clients
                weight = attentions(global_distribution, local_features, cosine_distance)
                weight_aggregate(weight, global_model, model_clients)
                batch_att_log = i*min_train_index+b_i*256
                temp_att_log = [batch_att_log,weight.numpy().tolist()]
                atten_log.append(temp_att_log)
                # if cmp_exp:
                #     avg_aggregate(global_model_avg,fl_avg_model_clients)

                # Update gusses model
                for j in range(n_clients):
                    gusses_opts[j].zero_grad()
                    loss_gene = loss_function_generation(global_distribution, gusses[j])
                    loss_gene.backward(retain_graph=True)
                    gusses_opts[j].step()
                    batch_gene = b_i*256+i*min_train_index
                    # print("[generator stat]epoch:{},batch:{},client:{},loss:{}".format(i,batch_gene,j,loss_gene))
                    temp_generator_log = [i,batch_gene,j,float(loss_gene.detach().numpy())]
                    generator_loss_client_log.append(temp_generator_log)



                if b_i % args.frequency_of_the_test == 0 or b_i == int(min_train_index / args.batch_size) - 1:
                    # if cmp_exp:
                    #     acc_avg,f1_avg,cm_avg = acc_f1(global_model_avg,test_feature,test_label,test_supply)
                    #     acc_att,f1_att,cm_att = acc_f1(global_model,test_feature,test_label,test_supply)
                    #     acc_single,f1_single,cm_single = acc_f1(single_model,test_feature,test_label,test_supply)
                    #     batch_test = i*min_train_index+b_i*args.batch_size
                    #     # print("[global-test]avg-fl batch:{},acc:{},f1:{}".format(batch_test,acc_avg, f1_avg))
                    #     # print("[global-test]att-fl batch:{},acc:{},f1:{}".format(batch_test,acc_att, f1_att))
                    #     # print("[global-test]single batch:{},acc:{},f1:{}".format(batch_test,acc_single, f1_single))
                    #     temp_avg = [batch_test,acc_avg,f1_avg]
                    #     temp_att = [batch_test, acc_att, f1_att]
                    #     temp_single = [batch_test, acc_single, f1_single]
                    #     test_att_log.append(temp_att)
                    #     test_avg_log.append(temp_avg)
                    #     test_single_log.append(temp_single)


                    acc_att,f1_att,cm_att = acc_f1(global_model,test_feature,test_label,test_supply,test_adj,args.model)

                    if f1_att > test_f1_best:
                        test_f1_best = f1_att
                    batch_test = i * min_train_index + b_i * args.batch_size
                    print("[gnn-test]batch:{},acc:{},f1:{}".format(batch_test,acc_att,f1_att))


        print(test_f1_best)

        return test_f1_best


if __name__ == "__main__":
    def add_args(parser):
        """
        parser : argparse.ArgumentParser
        return a parser added with args required by fit
        """
        # Training settings

        parser.add_argument('--case_name', type=str, default='knn', help='Dataset used for training')

        parser.add_argument('--model', type=str, default='gcn',
                            help='Model name. Currently supports SAGE, GAT and GCN.')

        parser.add_argument('--normalize_features', type=bool, default=False,
                            help='Whether or not to symmetrically normalize feat matrices')

        parser.add_argument('--normalize_adjacency', type=bool, default=False,
                            help='Whether or not to symmetrically normalize adj matrices')

        parser.add_argument('--sparse_adjacency', type=bool, default=False,
                            help='Whether or not the adj matrix is to be processed as a sparse matrix')

        parser.add_argument('--hidden_size', type=int, default=32, help='Size of GNN hidden layer')

        parser.add_argument('--node_embedding_dim', type=int, default=32,
                            help='Dimensionality of the vector space the atoms will be embedded in')

        parser.add_argument('--alpha', type=float, default=0.2, help='Alpha value for LeakyRelu used in GAT')

        parser.add_argument('--num_heads', type=int, default=2, help='Number of attention heads used in GAT')

        parser.add_argument('--dropout', type=float, default=0.3, help='Dropout used between GraphSAGE layers')

        parser.add_argument('--readout_hidden_dim', type=int, default=64, help='Size of the readout hidden layer')

        parser.add_argument('--graph_embedding_dim', type=int, default=64,
                            help='Dimensionality of the vector space the molecule will be embedded in')

        parser.add_argument('--client_optimizer', type=str, default='adam', metavar="O",
                            help='SGD with momentum; adam')

        parser.add_argument('--lr', type=float, default=0.0015, metavar='LR',
                            help='learning rate (default: 0.0015)')

        parser.add_argument('--batch_size', type=int, default=256, metavar='BS',
                            help='batch size (default: batch_size)')

        parser.add_argument('--wd', help='weight decay parameter;', metavar="WD", type=float, default=0.001)

        parser.add_argument('--epochs', type=int, default=10, metavar='EP',
                            help='how many epochs will be trained locally')

        parser.add_argument('--frequency_of_the_test', type=int, default=1, help='How frequently to run eval')

        parser.add_argument('--device', type=str, default="cpu", metavar="DV", help='device for training')

        parser.add_argument('--num_gnn_bp', type=int, default=8,
                            help='the number of gnn model update before one upload the server')

        parser.add_argument('--num_clients', type=int, default=5,
                            help='the number of clients')

        args = parser.parse_args()

        return args


    parser = argparse.ArgumentParser()
    args = add_args(parser)

    rlt = dict()
    for data in ['mlpMixer', 'vggTransformer']:
        for case_name in ['knn','pcc','plv']:
            for model in ['gps','sage','gat','gcn']:
                key = case_name+"_"+model+"_"+data
                print(key)
                path = {
                        'feature': "./output/feature_net/Feature_0_pytorch_{}.npz".format(data),
                        "disM": "./data/ISRUC_S3/DistanceMatrix.npy",
                        'save': "./result/"
                    }
                args.case_name = case_name
                args.model = model
                rlt[key] = run_fl_exp(args, path, Generator_global_distribution, run_vae, cmp_exp=False)
                print(rlt)
    print(rlt)
    with open("compare_experiment_rlt.json","w") as fp:
        fp.write(json.dumps(rlt))
