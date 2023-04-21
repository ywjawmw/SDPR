# -*- coding: utf-8 -*-
# @Time    : 2022/1/10 11:10
# @Author  :
# @File    : batch_test.py
# @Description : 测试核心代码
import time

from utils.parser import parse_args
from utils.load_dataset import *
import multiprocessing
import torch
import copy
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_curve


cores = multiprocessing.cpu_count() // 2

args = parse_args()
Ks = eval(args.Ks)
set_K = eval(args.set_K)
data_generator = DataSet(args=args, path=args.data_path + args.dataset, batch_size=args.batch_size)
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
BATCH_SIZE = args.batch_size


def test(model, users_to_test, test_group_list, drop_flag=False, repeat=1):
    """
    Args:
        users_to_test: list [test_num, n_user]  one-hot
        test_group_list:  list len=test_num, list[i] = ['sym-x_sym-y', [herb id set]]
        如果是计算大集合的话，test_group_list即data_generator.test_group_set_repeat,
            -- list[i] = ['sym-x_sym-y', [[herb id set1], [herb id set2],...]]
            -- eg: ['68', [
                            list：[251, 27, 45], [156, 89, 41, 59, 45], [144, 15, 44, 3, 5, 60, 50]
                          ]
                   ]
    """
    device = torch.device('cuda:' + str(args.gpu_id))
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)),
              'ndcg': np.zeros(len(Ks)), 'rmrr': np.zeros(len(Ks)), 'IOU': 0.0}
    # test_users = users_to_test
    test_users = torch.tensor(users_to_test, dtype=torch.float32).to(args.device)
    item_batch = range(ITEM_NUM)
    if drop_flag == False:
        user_embeddings, pos_i_g_embeddings = model(test_users, pos_items=item_batch, train=False,
                                                    user_padding_set=data_generator.test_users_padding)
    else:
        args.mess_dropout = eval(str(args.mess_dropout))
        user_embeddings, pos_i_g_embeddings = model(test_users, pos_items=item_batch, train=False,
                                                    user_padding_set=data_generator.test_users_padding)
        print('drop_flag: ', drop_flag, ',\t mess_dropout: ', args.mess_dropout)
    rate_batch = model.create_batch_rating(pos_i_g_embeddings, user_embeddings)
    print('rate_batch ', rate_batch.shape)

    user_batch_rating_uid = zip(test_users, rate_batch)
    user_rating_dict = {}

    index = 0
    for entry in user_batch_rating_uid:
        rating = entry[1]         # (1, 753)
        temp = [(i, float(rating[i])) for i in range(len(rating))]
        user_rating_dict[index] = temp
        index += 1
    # user_rating_dict {sym-1: [(herb1, rate), (herb2, rate), ..., (herb753, rate)], ...,
    #                   sym-1162:[(herb1, rate), (herb2, rate), ..., (herb753, rate)]}

    precision_n = np.zeros(len(Ks))
    recall_n = np.zeros(len(Ks))
    ndcg_n = np.zeros(len(Ks))
    rmrr_n = np.zeros(len(Ks))
    topN = Ks

    gt_count = 0
    candidate_count = 0
    if repeat == 1:
        for index in range(len(test_group_list)):
            entry = test_group_list[index]
            v_list = entry[1]  # sym-index's true herb set list
            rating = user_rating_dict[index]
            candidate_count += len(rating)
            rating.sort(key=lambda x: x[1], reverse=True)
            candidate_count += len(rating)
            K_max = topN[len(topN) - 1]
            for ii in range(len(topN)):  # topN: [5, 10, 15, 20]
                top_recall, top_precision, top_ndcg, top_rmrr, top_iou = 0., 0., 0., 0., 0.
                for v in v_list:  # v:对应的ground truth
                    r = []
                    for i in rating[:K_max]:
                        herb = i[0]
                        if herb in v:
                            r.append(1)
                        else:
                            r.append(0)
                    number = 0
                    herb_results = []  # 推荐列表中herb 集合
                    for i in rating[:topN[ii]]:
                        herb = i[0]
                        herb_results.append(herb)
                        if herb in v:
                            number += 1
                    herb_v = set(herb_results + v)
                    # todo: modified MRR to Rank-MRR
                    mrr_score = 0.
                    for a_rank in range(len(v)):  # herb 在grand truth中的位置a_rank
                        if v[a_rank] in herb_results:
                            a_refer = herb_results.index(v[a_rank])  # herb 在推荐列表中的位置a_refer
                            mrr_score += 1.0 / (abs(a_refer - a_rank) + 1)
                    if float(number / topN[ii]) > top_precision:  # 使用precision选择GT
                        top_precision = float(number / topN[ii])
                        top_recall = float(number / len(v))
                        top_ndcg = ndcg_at_k(r, topN[ii])
                        top_rmrr = mrr_score / len(v)
                precision_n[ii] = precision_n[ii] + top_precision  # [ii]所有测试数据top k的precision之和
                recall_n[ii] = recall_n[ii] + top_recall
                ndcg_n[ii] = ndcg_n[ii] + top_ndcg
                rmrr_n[ii] = rmrr_n[ii] + top_rmrr
    print('gt_count ', gt_count)
    print('candidate_count ', candidate_count)
    print('ideal candidate count ', len(test_group_list) * ITEM_NUM)
    for ii in range(len(topN)):
        result['precision'][ii] = precision_n[ii] / len(test_group_list)
        result['recall'][ii] = recall_n[ii] / len(test_group_list)
        result['ndcg'][ii] = ndcg_n[ii] / len(test_group_list)
        result['rmrr'][ii] = rmrr_n[ii] / len(test_group_list)
    return result


def range_test(model, model_stage1, users_to_test, items_to_test, drop_flag=False):
    """
    Args:
        model:  带有对比学习的第二阶段的model
        model_stage1: 第一阶段表示学习的model
        users_to_test: list[test_num, n_user]  one-hot
        items_to_test: list[test_num, n_item]  one-hot
    """
    test_users = torch.tensor(users_to_test, dtype=torch.float32).to(args.device)   # [TN, n_user]
    test_items = torch.tensor(items_to_test, dtype=torch.float32).to(args.device)  # [TN, n_item]
    item_padding_set, user_padding_set = data_generator.test_items_padding, data_generator.test_users_padding
    # user_embeddings: 症状集合的embedding
    # ia_embeddings: 每个中药的embedding
    # item_embeddings: 中药集合的embedding
    user_embeddings, _, ia_embeddings, item_embeddings, _ = \
        model(test_users, item_padding_set=item_padding_set, items=test_items,
              user_padding_set=user_padding_set)
    if model_stage1 != 1:
        user_embeddings_smgcn, _, ia_embeddings_smgcn, _, _ = \
            model_stage1(test_users, items=test_items)
    # logits_user_neg: 症状集合和该症状集合对应的所有top K集合的点积值 [TN, TK: topK 集合的个数]
    logits_user = torch.matmul(user_embeddings, item_embeddings.transpose(0, 1))  # 测试集中所有症状集合和中药集合的內积 [TN-3443, TN]
    # 计算MRR
    vals, indices = logits_user.sort(descending=True)
    mrr_score = 0.
    recall_list = np.zeros(len(set_K))  # [5, 10, 15, 20]
    pre_list = np.zeros(len(set_K))
    test_positive_list = data_generator.test_positive_list
    labels_gt = torch.tensor(test_positive_list).to(args.device)
    labels_gt = labels_gt.contiguous().view(-1, 1)  # [TN, 1] 列向量
    mask = torch.eq(labels_gt, labels_gt.T).float().to(args.device)  # [TN, TN]     1表示正例， 0表示负例
    # 计算recall和MRR值
    for test_index in range(logits_user.size(0)):
        posit_index = (mask[test_index] == 1.).nonzero().squeeze(dim=1).to(args.device)  #
        score = 0.
        for pi in posit_index:
            range_index = (indices[test_index] == int(pi)).nonzero().squeeze(dim=1).to(args.device)  #
            mrr = torch.reciprocal(range_index + 1)  # mrr
            if mrr >= 1 / posit_index.size(0):
                mrr = 1.
            score += mrr
        score = score / posit_index.size(0)
        mrr_score += score
        # 计算Recall值
        for ii in range(len(set_K)):
            k = set_K[ii]  #  5、10、15、20
            rec_id = indices[test_index][:k]  # K id
            number = 0
            for pi in posit_index:  # ground truth id
                if pi in rec_id:
                    number += 1
            recall_list[ii] = recall_list[ii] + number / posit_index.size(0)
            pre_list[ii] = pre_list[ii] + number / k
    mrr_score = mrr_score / mask.size(0)  # 平均
    for ii in range(len(set_K)):
        recall_list[ii] = recall_list[ii] / mask.size(0)  #
        pre_list[ii] = pre_list[ii] / mask.size(0)
    return mrr_score.item(), recall_list, pre_list


def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
    """
    # dcg_max = dcg_at_k(np.ones_like(r), k, method)
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


