import torch.utils.data
from torch import optim
from torch.nn import functional as F
from utlis.vae_utlis import get_data_feature,loss_function_generation
from utlis.inter_intra_utlis import *
import pandas as pd


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

def fl_gnn_split_data(feature,label,supply,n_clients):
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
        clients_data.append((feature[mask], label[mask], supply[mask]))

    for i in range(n_clients):
        print(clients_data[i][1].sum(axis=0))
    return clients_data

def get_gnn_raw_data(path,n_clients,win_size,split_ratio = 0.8):
    data_process = DataProcess(path)
    feature, label = data_process.read_data()
    n_sample = len(feature)
    train_index = int(n_sample*split_ratio)
    feature = np.array(feature)
    feature_train = feature[:train_index]
    label_train = label[:train_index]

    feature_test  = feature[train_index:]
    label_test = label[train_index:]

    supplement_sage_gps_train = adpator_sage_gps(feature_train)
    supplement_sage_gps_test = adpator_sage_gps(feature_test)

    supplement_sage_gps_train =  np.array(supplement_sage_gps_train)
    clients_data = fl_gnn_split_data(feature_train,label_train,supplement_sage_gps_train,n_clients)
    feature_test_align,label_test_align,supplement_sage_gps_test_align = \
        data_process.align_data(feature_test,label_test,supplement_sage_gps_test,win_size)
    test_data = (feature_test_align,label_test_align,supplement_sage_gps_test_align)
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

def fl_model_data_initialization(path,Model,args_model,args,n_clients):
    # clients_data partially non-iid
    clients_data,test_data = get_gnn_raw_data(path,n_clients,args_model['win_size'],0.9)
    featrue_align_clients, label_clients, model_clients, opt_clients = [], [], [], []
    supplement_clients = []
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    n_clients = len(clients_data)
    for i in range(n_clients):
        feature, label, supply = clients_data[i]
        supplement_sage_gps = adpator_sage_gps(feature)
        featrue_align, label_align,supplement_sage_gps_align = DataProcess.align_data(feature, label,supplement_sage_gps,args_model['win_size'])
        featrue_align_clients.append(featrue_align)
        label_clients.append(label)
        supplement_clients.append(supplement_sage_gps_align)
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
        model = Model(**args_model)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        model_clients.append(model)
        opt_clients.append(opt)
    # test_data_clients, test_label_clients, test_supplement_clients,test_global_data,test_global_label,test_global_supply \
    #     = test_data_process(test_data_clients,test_label_clients,test_supplement_clients)
    return featrue_align_clients, label_clients, model_clients, opt_clients,supplement_clients,criterion,test_data

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

def run_fl_exp(args,args_model,path,Model,Generator_global_distribution,run_vae,cmp_exp = False):
        device = args.device
        n_clients = args.num_clients

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
        feature_align_clients, label_clients, model_clients, opt_clients, supplement_clients, criterion, test_data\
            = fl_model_data_initialization(path, Model, args_model, args, n_clients)

        # initialization of vae
        if not os.path.exists("log/VAE/Model/VAE_Client1.pth.tar"):
            run_vae(clients_data_vae, clients_valid_vae)

        local_features = get_distribution_repr(clients_data_vae)
        # print(local_repr)

        # initialization of gusses model
        gusses_model, gusses_opts = gusses_model_initialization(Generator_global_distribution, n_clients)

        # initialization of global model
        global_model = Model(**args_model)

        # add compare experiment
        if cmp_exp:
            single_model,single_opt,fl_avg_model_clients,fl_avg_opt_clients,global_model_avg = \
                add_cmp_exper(Model,args_model,args,n_clients)

        # trainning process

        test_f1_best = -1
        print("win size = {}".format(args_model['win_size']))

        num_gnn_bp = args.num_gnn_bp

        train_len_clients = [len(ele) for ele in feature_align_clients]
        test_feature,test_label,test_supply = test_data
        for i in range(args.epochs):
            print("epoch:{}".format(i))
            # gusses = []
            # global_distribution = torch.zeros_like(local_features[0])

            # prepare client data
            train_feature_clients = []
            train_label_clients = []
            train_supply_clients = []
            for j in range(n_clients):
                # shuffle = shuffle_clients[j]
                # train_index = train_index_clients[j]
                train_label = label_clients[j]
                train_feature = feature_align_clients[j]
                train_supply = supplement_clients[j]

                train_feature,train_label,train_supply = shuffle_train(train_feature,train_label,train_supply)

                train_label = iter(train_label)
                train_feature = iter(train_feature)
                train_supply = iter(train_supply)
                # model = model_clients[j]
                # opt = opt_clients[j]

                # train_feature = iter(featrue_align[shuffle[:train_index]])
                # train_lable = iter(label[shuffle[:train_index]])
                # train_supply = iter(supply[shuffle[:train_index]])
                train_feature_clients.append(train_feature)
                train_label_clients.append(train_label)
                train_supply_clients.append(train_supply)

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
                        if cmp_exp:
                            model = model_clients[client_index]
                            opt_att = opt_clients[client_index]
                            model_avg = fl_avg_model_clients[client_index]
                            opt_avg = fl_avg_opt_clients[client_index]

                            optimizer_att = opt_att
                            optimizer_avg = opt_avg
                            optimizer_single = single_opt

                            optimizer_att.zero_grad()
                            optimizer_avg.zero_grad()
                            optimizer_single.zero_grad()

                            batch_loss_att,batch_loss_avg,batch_loss_single = \
                                calculate_loss_cmp(model,model_avg,single_model,train_feature, train_label, \
                                                   train_supply, args.num_gnn_bp,criterion)
                            batch_loss_att.backward()
                            batch_loss_avg.backward()
                            batch_loss_single.backward()

                            optimizer_att.step()
                            optimizer_avg.step()
                            optimizer_single.step()

                            batch_num = i*min_train_index +256 * b_i + 8 * _
                            # print("batch:{},loss_att:{},loss_avg:{},loss_single:{}".format(batch_num,batch_loss_att,\
                            #                                                                batch_loss_avg,batch_loss_single))
                            temp_train_log = [i, batch_num, client_index, float(batch_loss_att.detach().numpy())]
                            train_loss_client_log.append(temp_train_log)
                        else:
                            model = model_clients[client_index]
                            opt = opt_clients[client_index]
                            optimizer = opt
                            optimizer.zero_grad()
                            batch_loss = calculate_loss(model, train_feature, train_label, train_supply, args.num_gnn_bp,
                                                        criterion)
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
                if cmp_exp:
                    avg_aggregate(global_model_avg,fl_avg_model_clients)

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
                    if cmp_exp:
                        acc_avg,f1_avg,cm_avg = acc_f1(global_model_avg,test_feature,test_label,test_supply)
                        acc_att,f1_att,cm_att = acc_f1(global_model,test_feature,test_label,test_supply)
                        acc_single,f1_single,cm_single = acc_f1(single_model,test_feature,test_label,test_supply)
                        batch_test = i*min_train_index+b_i*args.batch_size
                        # print("[global-test]avg-fl batch:{},acc:{},f1:{}".format(batch_test,acc_avg, f1_avg))
                        # print("[global-test]att-fl batch:{},acc:{},f1:{}".format(batch_test,acc_att, f1_att))
                        # print("[global-test]single batch:{},acc:{},f1:{}".format(batch_test,acc_single, f1_single))
                        temp_avg = [batch_test,acc_avg,f1_avg]
                        temp_att = [batch_test, acc_att, f1_att]
                        temp_single = [batch_test, acc_single, f1_single]
                        test_att_log.append(temp_att)
                        test_avg_log.append(temp_avg)
                        test_single_log.append(temp_single)


                    acc_att,f1_att,cm_att = acc_f1(global_model,test_feature,test_label,test_supply)

                    if f1_att > test_f1_best:
                        test_f1_best = f1_att
                    batch_test = i * min_train_index + b_i * args.batch_size
                    print("[gnn-test]batch:{},acc:{},f1:{}".format(batch_test,acc_att,f1_att))

                    # for j in range(n_clients):
                    #     test_data = test_data_clients[j]
                    #     test_label = test_label_clients[j]
                    #     test_supply = test_supplement_clients[j]
                    #     model = model_clients[j]
                    #
                    #     acc, f1, cm = acc_f1(model, test_data, test_label, test_supply)
                    #     batch_test = i*min_train_index+b_i*args.batch_size
                    #     # print("[gnn-test]client:{},acc:{},f1:{}".format(j,acc, f1))
                    #     temp_test_log = [i,batch_test,j,acc,f1]
                    #     test_metric_client_log.append(temp_test_log)
                    #     if f1 > test_f1_best:
                    #         test_f1_best = f1

        print(test_f1_best)
        # train_log_df = pd.DataFrame(train_loss_client_log,columns=["Epoch","batch","clients","loss"])
        # test_log_df  = pd.DataFrame(test_metric_client_log,columns=["Epoch","batch","clients","acc","f1"])
        # gene_log_df  = pd.DataFrame(generator_loss_client_log,columns=["Epoch","batch","clients","loss"])
        # atten_log_df = pd.DataFrame(atten_log,columns=["batch","att"])
        #
        # train_log_df.to_csv(path['log']+"train_log_1.csv",index = False)
        # test_log_df.to_csv(path['log']+"test_log_1.csv",index=False)
        # gene_log_df.to_csv(path['log']+"gene_log_1.csv",index = False)
        # atten_log_df.to_csv(path['log']+"att_log.csv",index = False)

        if cmp_exp:
            test_att_log_df = pd.DataFrame(test_att_log, columns=["batch","acc","f1"])
            test_avg_log_df = pd.DataFrame(test_avg_log, columns=["batch", "acc", "f1"])
            test_single_log_df = pd.DataFrame(test_single_log, columns=["batch", "acc", "f1"])
            test_att_log_df.to_csv(path['log'] + "fl_att.csv", index=False)
            test_avg_log_df.to_csv(path['log'] + "fl_avg.csv", index=False)
            test_single_log_df.to_csv(path['log'] + "fl_single.csv", index=False)

        return test_f1_best


if __name__ == "__main__":
    path = "../output/feature_net/mlpmixer/data/Feature_1.npz"
    n_clients = 5
    train_feature, val_feature, train_targets, val_targets = get_raw_data(path)
    train_feature, val_feature = flatten_data(train_feature,val_feature)
    clients_data, clients_valid  = split_data(train_feature,train_targets,val_feature,val_targets,n_clients)
    clients_data, clients_valid = max_scale(clients_data, clients_valid)
