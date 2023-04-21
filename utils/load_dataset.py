import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import os
from sklearn.preprocessing import MultiLabelBinarizer
from utils.parser import parse_args
import torch

#  fix seed
args1 = parse_args()
seed = args1.seed
print("use seed：", seed)
rd.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class DataSet(object):
    def __init__(self, args, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        train_file = path + '/' + args.train_file
        test_file = path + '/' + args.test_file

        sym_pair_file = path + '/symPair-5.txt'
        herb_pair_file = path + '/herbPair-40.txt'

        # get number of users and items
        self.n_users, self.n_sets, self.n_items = self.load_data_size(args, '{}_data_size.txt')[:3]
        self.n_train, self.n_test = 0, self.load_data_num(args, '{}_num.txt')[0]

        # herbs in train
        self.train_items = set()
        # herbs in test
        self.test_items = set()
        self.test_all_users = set()
        self.all_items = set()
        # prescriptions in train
        self.train_pres = list()
        self.max_item_len = self.load_data_size(args, '{}_item_length.txt')[0]

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

        self.test_group_set = list()
        self.test_group_set_repeat = list()   #
        self.test_users = np.zeros((self.n_test, self.n_users), dtype=float)
        self.test_items_hot = np.zeros((self.n_test, self.n_items), dtype=float)
        self.item_weights = np.zeros((self.n_items, 1), dtype=float)

        self.train_set_list = list()  # herb id list  [[]]

        self.epoch = args.epoch

        user_item_count = 0

        with open(train_file) as f:
            self.train_fang = dict()     # key: symptom set value：all herb list
            self.train_symset_herbset = dict()    # key: symptom set value: [[]] herb list
            for l in f.readlines():
                if len(l) > 0:
                    temp = l.strip().split('\t')
                    tempS = temp[0].split(" ")
                    uids = [int(i) for i in tempS]
                    tempH = temp[1].split(' ')
                    items = [int(i) for i in tempH]
                    self.train_symset_herbset.setdefault(str(uids), []).append(items)
                    if str(uids) in self.train_fang.keys():
                        self.train_fang[str(uids)] = list(set(self.train_fang[str(uids)] + items))
                    else:
                        self.train_fang[str(uids)] = items
                    if items not in self.train_set_list:
                        self.train_set_list.append(items)

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    temp = l.strip().split('\t')
                    tempS = temp[0].split(" ")
                    tempH = temp[1].split(" ")
                    try:
                        uids = [int(i) for i in tempS]
                        items = [int(i) for i in tempH]
                        self.train_pres.append([uids, items])
                        for item in items:
                            self.train_items.add(item)
                            self.all_items.add(item)
                            self.item_weights[item][0] += 1
                        for user in uids:
                            for item in items:
                                if self.R[user, item] != 1.:
                                    self.R[user, item] = 1.
                                    user_item_count += 1
                    except Exception:
                        continue
                    self.n_train += 1

            print('# user-item count ', user_item_count)

            print('item_weight ', len(self.item_weights))
            item_freq_max = self.item_weights.max()
            print('item_freq: item_weight.shape[0] and [1]', self.item_weights.shape[0], ' ',
                  self.item_weights.shape[1])
            for index in range(self.item_weights.shape[0]):
                self.item_weights[index][0] = item_freq_max * 1.0 / self.item_weights[index][0]

        test_index = 0
        self.test_positive_list = list()
        with open(test_file) as f:
            self.test_fang = dict()
            index = 0
            fang_index = dict()
            for l in f.readlines():
                if len(l) > 0:
                    temp = l.strip().split('\t')
                    tempS = temp[0]
                    tempH = temp[1].split(' ')
                    items = [int(i) for i in tempH]
                    self.test_fang.setdefault(tempS, []).append(items)
                    if str(tempS) in fang_index.keys():
                        self.test_positive_list.append(fang_index[str(tempS)])
                    else:
                        self.test_positive_list.append(index)
                        fang_index[str(tempS)] = index
                    index += 1

        with open(test_file) as f:
            self.test_users_padding = list()
            self.test_items_padding = list()
            for l in f.readlines():
                if len(l) > 0:
                    if len(l) == 0: break
                    l = l.strip('\n')
                    temp = l.strip().split('\t')
                    tempS = temp[0].split(' ')
                    tempH = temp[1].split(' ')
                    uids = [int(i) for i in tempS]
                    try:
                        for uid in uids:
                            self.test_users[test_index][uid] = 1.
                            self.test_all_users.add(uid)
                        items = [int(i) for i in tempH]
                        item_padding = [self.n_items] * (self.max_item_len - len(items))
                        padding_items = items + item_padding
                        self.test_items_padding.append(padding_items)
                        for item in items:
                            self.test_items.add(item)
                            self.all_items.add(item)
                            self.test_items_hot[test_index][item] = 1.    # test ground truth 中 herb的multi hot
                    except Exception:
                        continue
                    test_index += 1
                    uid, test_items = uids, items
                    user_index = ''
                    for user in uid:
                        user_index += str(user) + "_"
                    user_index = user_index[:-1]
                    self.test_group_set.append([user_index, test_items])
                    self.test_group_set_repeat.append([user_index, self.test_fang[temp[0]]])
                    # self.n_test += 1
            print("#multi-hot for test users\t", len(self.test_users))
            print("#test\t", len(self.test_group_set))
            print("test max item len:\t", self.max_item_len)
        self.print_statistics()

        self.sym_pair = sp.dok_matrix((self.n_users, self.n_users), dtype=np.float32)
        self.herb_pair = sp.dok_matrix((self.n_items, self.n_items), dtype=np.float32)
        sym_pair_count = 0
        with open(sym_pair_file) as f_sym_pair:
            for l in f_sym_pair.readlines():
                if len(l) == 0: break
                pair = l.strip().split(' ')
                sym1 = int(pair[0])
                sym2 = int(pair[1])
                # print('sym-pair ', sym1, ' ', sym2)
                self.sym_pair[sym1, sym2] = 1.
                self.sym_pair[sym2, sym1] = 1.
                sym_pair_count += 2

        print('# sym pairs ', sym_pair_count)

        herb_pair_count = 0
        with open(herb_pair_file) as f_herb_pair:
            for l in f_herb_pair.readlines():
                if len(l) == 0: break
                pair = l.strip().split(' ')
                herb1 = int(pair[0])
                herb2 = int(pair[1])
                # print('herb ', herb1, ' ', herb2)
                self.herb_pair[herb1, herb2] = 1.
                self.herb_pair[herb2, herb1] = 1.
                herb_pair_count += 2

        print('#herb pairs ', herb_pair_count)

    def load_data_size(self, args, name):
        with open(os.path.join(self.path, name.format(args.dataset)), 'r') as f:
            return [int(s) for s in f.readline().split('\t')]

    def load_data_num(self, args, name):
        with open(os.path.join(self.path, name.format(args.test_file.replace(".txt", ""))), 'r') as f:
            return [int(s) for s in f.readline().split('\t')]

    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            sym_pair_mat = sp.load_npz(self.path + '/s_sym_pair_mat.npz')
            herb_pair_mat = sp.load_npz(self.path + '/s_herb_pair_mat.npz')
            print('already load sym_pair adjacency matrix', sym_pair_mat.shape)
            print('already load herb_pair adjacency matrix', herb_pair_mat.shape)
            print('already load adj matrix', adj_mat.shape, time() - t1)
        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat, sym_pair_mat, herb_pair_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)
            sp.save_npz(self.path + '/s_sym_pair_mat.npz', sym_pair_mat)
            sp.save_npz(self.path + '/s_herb_pair_mat.npz', herb_pair_mat)
        return adj_mat, norm_adj_mat, mean_adj_mat, sym_pair_mat, herb_pair_mat

    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        # 双向的原始邻接矩阵
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, 'time:', time() - t1)
        t2 = time()

        sym_pair_adj_mat = self.sym_pair.tolil().todok()
        print('already create sym_pair adjacency matrix', sym_pair_adj_mat.shape, 'time:', time() - t2)
        t3 = time()

        herb_pair_adj_mat = self.herb_pair.tolil().todok()
        print('already create herb_pair adjacency matrix', herb_pair_adj_mat.shape, 'time:', time() - t3)

        def normalized_adj_single(adj):
            # 行归一化  每行的行sum列表
            rowsum = np.array(adj.sum(1))
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            # 每一个元素都除上该行的sum
            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def normalized_adj_bi(adj):
            rowsum = np.array(adj.sum(1))
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
            bi_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
            return bi_adj

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))   # sp.eye单位矩阵
        mean_adj_mat = normalized_adj_single(adj_mat)
        print('already normalize adjacency matrix', time() - t2)

        print('sym_pair 和 herb_pair 有 self-connection, sum!!')
        sym_pair_adj_mat = sym_pair_adj_mat + sp.eye(sym_pair_adj_mat.shape[0])
        herb_pair_adj_mat = herb_pair_adj_mat + sp.eye(herb_pair_adj_mat.shape[0])

        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr(), sym_pair_adj_mat.tocsr(), herb_pair_adj_mat.tocsr()

    def sample(self, args):
        self.train_items_padding = list()
        self.train_users_padding = list()
        self.train_positive_list = list()
        fang_sample = dict()
        fang_items = dict()
        sample_ids = [i for i in range(len(self.train_pres))]
        # max_batch_item_len = 0
        if self.batch_size <= len(sample_ids):
            pres_ids = rd.sample(sample_ids, self.batch_size)    # sample batch size data
        else:
            pres_ids = [rd.choice(sample_ids) for _ in range(self.batch_size)]

        users = []
        for pres_id in pres_ids:
            users.append(self.train_pres[pres_id])      # users: [[[user id set], [item id set]], ...]
            # if len(self.train_pres[pres_id][1]) > self.max_item_len:
            #     self.max_item_len = len(self.train_pres[pres_id][1])
        self.data_sample_ids = pres_ids
        user_sets = np.zeros((len(users), self.n_users), dtype=float)    # multi-hot [B, n_users]
        item_sets = np.zeros((len(users), self.n_items), dtype=float)    # multi-hot [B, n_items]
        self.item_sets_repeat = np.zeros((len(users), self.n_items), dtype=float)  # multi-hot [B, n_items]
        # self.item_sets_repeat_dataset = np.zeros((len(users), self.n_items), dtype=float)  # multi-hot [B, n_items]
        user_set = set()
        item_set = set()
        for index in range(len(users)):
            uids = users[index][0]
            items = users[index][1]
            padding = [self.n_items] * (self.max_item_len - len(items))
            padding_items = items + padding
            self.train_items_padding.append(padding_items)   #
            if str(uids) in fang_sample.keys():
                self.train_positive_list.append(fang_sample[str(uids)])
            else:
                self.train_positive_list.append(index)
                fang_sample[str(uids)] = index
            if str(uids) in fang_items.keys():
                fang_items[str(uids)] = list(set(fang_items[str(uids)] + items))
            else:
                fang_items[str(uids)] = items
            # self.train_items_len.append([1] * len(items) + [0] * (self.max_item_len - len(items)))
            for uid in uids:
                user_sets[index][int(uid)] = 1.    # multi-hot
                user_set.add(int(uid))
            for item in items:
                item_sets[index][int(item)] = 1.    # multi-hot
                item_set.add(int(item))
        herb_sets_list = list()
        for index in range(len(users)):
            uids = users[index][0]
            if args.all_dataset == 0:
                herb_list = fang_items[str(uids)]
            else:
                herb_list = self.train_fang[str(uids)]
            herb_sets_list.append(herb_list)
        self.item_sets_repeat = MultiLabelBinarizer(classes=range(0, self.n_items)).fit_transform(herb_sets_list)    # [B, herb_num]
        self.item_sets_repeat = np.array(self.item_sets_repeat, dtype=float)
        return user_sets, list(user_set), item_sets, list(item_set)

    def print_statistics(self):
        print('symptom n_users=%d, herb n_items=%d' % (self.n_users, self.n_items))
        print('#train herb  train_items %d' % (len(self.train_items)))
        print('#test herb  test_items %d' % (len(self.test_items)))
        print('#test syn  test_all_users %d' % (len(self.test_all_users)))
        print('#all herb: all_items %d' % (len(self.all_items)))
        print('item_max_len:\t', self.max_item_len)
        print('***********************para********************************')
        print('lr:', args1.lr)
        print('regs:', str(args1.regs))
        print('batch_size:', str(args1.batch_size))
        print('seed:', args1.seed)
        print('step：', args1.step)
        print('t=?', args1.t)
        print('co_lamda?', args1.co_lamda)



