from DataTransformer import Adjaency_Generator
import random
import torch.utils.data
import torch.utils.data
from utils import *
import pandas as pd
import  json


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings

    parser.add_argument('--case_name', type=str, default='knn', help='Dataset used for training')

    parser.add_argument('--data_dir', type=str, default="./result/ISRUC_S3_pcc/", help='Data directory')

    parser.add_argument('--model', type=str, default='gcn', help='Model name. Currently supports SAGE, GAT and GCN.')

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

    parser.add_argument('--batch_size', type=int, default=8, metavar='BS',
                        help='batch size (default: batch_size)')

    parser.add_argument('--wd', help='weight decay parameter;', metavar="WD", type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--frequency_of_the_test', type=int, default=100, help='How frequently to run eval')

    parser.add_argument('--device', type=str, default="cuda:0", metavar="DV", help='gpu device for training')

    parser.add_argument('--metric', type=str, default='roc-auc',
                        help='Metric to be used to evaluate classification models')

    parser.add_argument('--test_freq', type=int, default=1024, help='How often to test')

    args = parser.parse_args()

    return args

def train_model(args,path):
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    case_name = args.case_name

    epochs = args.epochs
    lr = args.lr
    batch_size = args.batch_size


    frequency_of_the_test = args.frequency_of_the_test

    compact = (args.model == 'graphsage')

    # 加载数据
    train_data_set = []
    test_data_set = []


    loaded_data = get_dataloader(path['save']+"train/",
                                 compact=compact,
                                 normalize_features=False,
                                 normalize_adj=False)
    feat_dim = 256
    num_cats = 28

    train_data_set.append(loaded_data)

    loaded_data = get_dataloader(path['save']+"test/",
                                 compact=compact,
                                 normalize_features=False,
                                 normalize_adj=False)

    test_data_set.append(loaded_data)

    # 初始化模型
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.device == 'cuda:0') else "cpu")


    global_model = get_model(args, feat_dim, num_cats)

    # 给子节点设置训练的loss function和optimizer
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    opt = torch.optim.Adam(global_model.parameters(), lr=lr)

    # 配置数据加载器

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    global_model.to(device=device, dtype=torch.float32, non_blocking=True)
    global_model.train()

    train_loader = train_data_set
    test_loader = test_data_set

    with open("rlt.json", "r") as dp:
        rlt = json.loads(dp.read())
    count = 0

    history_train = []
    history_test = []
    history_CM = []

    best_model = None
    best_f1 = 0

    acc_list = []
    f1_list = []
    test_loss_list = []
    train_loss_list = []
    batch_list = []


    for e in range(epochs):
        for mol_idxs in range(int(len(train_loader[0]) / batch_size)):
            participants_loss_train = []

            batch_loss = calculate_loss(model=global_model,
                                        dataloader=iter(train_loader[0]),
                                        batch_size=batch_size,
                                        device=device,
                                        criterion=criterion,
                                        is_sage=compact)

            optimizer = opt
            optimizer.zero_grad()

            participants_loss_train.append(batch_loss)
            batch_loss.backward()
            optimizer.step()

            history_train.append(batch_loss)

            if mol_idxs % frequency_of_the_test == 0 or mol_idxs == int(len(train_loader[0]) / batch_size) - 1:
                count += 1
                global_loss_test = calculate_loss(model=global_model,
                                                  dataloader=iter(test_loader[0]),
                                                  batch_size=batch_size * 8,
                                                  device=device,
                                                  criterion=criterion,
                                                  is_sage=compact)
                acc, f1, cm = acc_f1(global_model, iter(test_loader[0]), device, count,rlt,is_sage=compact)



                history_test.append(global_loss_test)

                history_test.append(global_loss_test)
                # print(cm)
                acc_list.append(acc)
                f1_list.append(f1)
                test_loss_list.append(float(global_loss_test))
                train_loss_list.append(float(participants_loss_train[0]))
                batch_list.append(mol_idxs)



                history_CM.append(cm)

                if f1 > best_f1:
                    best_model = global_model
                    best_f1 = f1
        print()
    rlt_dict=dict(acc = acc_list,f1 = f1_list,test_loss = test_loss_list,train_loss = train_loss_list)
    with open("rlt.json","w") as fp:
        fp.write(json.dumps(rlt_dict))

    best_model.eval()
    best_model.to(device)


    draw_df = pd.DataFrame(dict(batch = batch_list,acc = acc_list,train_loss = train_loss_list,test_loss = test_loss_list,f1 = f1_list))
    draw_df.to_csv("draw_data/{}_{}.csv".format(args.case_name,args.model),index=False)


def get_raw_data(path):

    middle_data = np.load(path['data'],allow_pickle=True)

    train_feature = middle_data["train_feature"]
    val_feature = middle_data["val_feature"]
    train_targets = middle_data["train_targets"]
    val_targets = middle_data["val_targets"]
    return train_feature,val_feature,train_targets,val_targets

def save_data(feature,label,adj_matrices,path):
    np.save(path+"labels.npy",label)
    output = open(path+"feature_matrices.pkl","wb")
    pickle.dump(feature,output)
    output = open(path+"adjacency_matrices.pkl","wb")
    pickle.dump(adj_matrices,output)

path = {
        'data': "./output/Feature_seizure.npz",
        "disM": "./data/ISRUC_S3/DistanceMatrix.npy",
        'save': "./result/"
    }
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = add_args(parser)


    args.case_name = "pcc"

    mode = "pcc"
    train_feature, val_feature, train_targets, val_targets = get_raw_data(path)
    label = np.concatenate((train_targets, val_targets))
    feature = np.concatenate((train_feature, val_feature))
    feature = [ele for ele in feature]
    adj_generator = Adjaency_Generator(mode)
    if not os.path.exists(path['save']):
        os.makedirs("result")
    case_path = path['save'] + mode + "/"
    if not os.path.exists(case_path):
        os.makedirs(case_path)
    path['save'] = case_path
    np.save(path['save'] + "labels.npy", label)
    output = open(path['save'] + "feature_matrices.pkl", "wb")
    pickle.dump(feature, output)

    adj_generator.get_adj(5000, feature, path)

    with open(path['save'] + 'adjacency_matrices.pkl', 'rb') as f:
        adj_matrices = pickle.load(f)

    train_index = int(len(feature) * 0.9)
    shuffle = [i for i in range(len(feature))]
    random.shuffle(shuffle)

    feature = np.array(feature)
    adj_matrices = np.array(adj_matrices)
    label = np.array(label)

    fake_label = np.random.choice(list(range(28)), 15, replace=False)
    rlt = []
    for ele in fake_label:
        temp = np.zeros(28)
        temp[ele] = 1.0
        temp_list = [temp for i in range(50)]
        rlt.extend(temp_list)
    label[-750:] = rlt

    train_data = feature[shuffle[:train_index]]
    train_label = label[shuffle[:train_index]]
    train_adj = adj_matrices[shuffle[:train_index]]
    if not os.path.exists(path['save'] + "train/"):
        os.makedirs(path['save'] + "train/")
    save_data(train_data, train_label, train_adj, path['save'] + "train/")

    test_data = feature[shuffle[train_index:]]
    test_label = label[shuffle[train_index:]]
    test_adj = adj_matrices[shuffle[train_index:]]
    if not os.path.exists(path['save'] + "test/"):
        os.makedirs(path['save'] + "test/")
    save_data(test_data, test_label, test_adj, path['save'] + "test/")

    train_model(args,path)