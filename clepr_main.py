# -*- coding: utf-8 -*-
# @Time    : 2022/1/17 16:22
# @Author  :
# @File    : clepr_main.py
# @Description :

import sys

from torch import nn

from model.CLEPR import CLEPR
from model.Attention_layer import SelfAttention
import datetime
from utils.helper import *
from utils.batch_test import *
import torch.optim as optim
from tensorboardX import SummaryWriter

def load_pretrained_data():
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, 'embedding')
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained embeddings.')
    except Exception:
        pretrain_data = None
    return pretrain_data


if __name__ == '__main__':
    startTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('start ', startTime)
    print('************CLEPR*************** ')
    print('result_index ', args.result_index)
    os.environ['CUDA_LAUNCH_BLOCKING'] = str(args.gpu_id)
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.device = torch.device('cuda:' + str(args.gpu_id))

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """
    plain_adj, norm_adj, mean_adj, sym_pair_adj, herb_pair_adj = data_generator.get_adj_mat()
    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)
    print("mess_dropout: ", args.mess_dropout)


    if args.adj_type == 'plain':
        config['norm_adj'] = plain_adj
        print('use the plain adjacency matrix')
        config['sym_pair_adj'] = sym_pair_adj
        config['herb_pair_adj'] = herb_pair_adj


    elif args.adj_type == 'norm':
        config['norm_adj'] = norm_adj
        print('use the normalized adjacency matrix')
        config['sym_pair_adj'] = sym_pair_adj
        config['herb_pair_adj'] = herb_pair_adj


    elif args.adj_type == 'gcmc':
        config['norm_adj'] = mean_adj
        print('use the gcmc adjacency matrix')
        config['sym_pair_adj'] = sym_pair_adj
        config['herb_pair_adj'] = herb_pair_adj

    else:
        config['norm_adj'] = mean_adj + sp.eye(mean_adj.shape[0])
        print('use the mean adjacency matrix')
        config['sym_pair_adj'] = sym_pair_adj
        config['herb_pair_adj'] = herb_pair_adj

    t0 = time()

    if args.pretrain == -1:
        pretrain_data = +load_pretrained_data()
    else:
        pretrain_data = None

    print('data generator size of {} b'.format(sys.getsizeof(data_generator)))
    model = CLEPR(data_config=config, pretrain_data=pretrain_data).to(args.device)

    """
    *********************************************************
    Save the model parameters.
    """

    if args.save_flag == 1:
        layer = '-'.join([str(l) for l in eval(args.layer_size)])
        mess_dr = '-'.join([str(l) for l in eval(str(args.mess_dropout))])
        weights_save_path = '%sweights-CLEPR/%s/%s/%s/%s/l%s_r%s_messdr%s/date_%s_%s/' % (
            args.weights_path, args.dataset, model.model_type, layer, args.embed_size,
            str(args.lr), '-'.join([str(r) for r in eval(args.regs)]), mess_dr,
            datetime.datetime.now().strftime('%Y-%m-%d'),
            args.save_tail)
        ensureDir(weights_save_path)

    cur_best_pre_0, stopping_step = 0, 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    """
    *********************************************************
    Reload the pretrained model parameters.
    """
    print("args.pretrain\t", args.pretrain)
    if args.pretrain == 1:
        print("pretrain==1")
        layer = '-'.join([str(l) for l in eval(args.layer_size)])
        mess_dr = '-'.join([str(l) for l in eval(str(args.mess_dropout))])
        # stage1 model (pretrain)
        weights_save_path_smgcn = '%sweights-CLEPR/%s/%s/%s/%s/%s/' % (
            args.weights_path, args.dataset, model.model_type, layer, args.embed_size,
            args.model_wpath
        )
        for save_id in range(1):
            weights_save_path = weights_save_path_smgcn + 'l%s_r%s_messdr%s_t%s_cor_%s_step%s_max_step_len_%s/%s/%s/' % (
                str(args.lr), '-'.join([str(r) for r in eval(args.regs)]), mess_dr,
                str(args.t),
                str(args.co_lamda),
                str(args.step),
                str(args.max_step_len),
                args.two_save_tail,
                str(save_id)
            )
            pretrain_path = weights_save_path
            print('load the pretrained model parameters from: ', pretrain_path)
            model = torch.load(weights_save_path + 'model.pkl', map_location=lambda storage, loc: storage)
            if model:
                print("start to load pretrained model")
                model = model.to(args.device)
                print('load the embedding model parameters from: ', weights_save_path_smgcn)
                model_smgcn = torch.load(weights_save_path_smgcn + 'model.pkl', map_location=lambda storage, loc: storage).to(
                    args.device)
                if args.report != 1:
                    ret = test(model, list(data_generator.test_users),
                               data_generator.test_group_set_repeat, drop_flag=True, repeat=1)  # 测试使用集合列表
                    cur_best_pre_0 = ret['precision'][0]
                    pretrain_ret = "pretrained model \trecall=[%s], precision=[%s],  ndcg=[%s], rmrr=[%s]" % \
                                 ('\t'.join(['%.5f' % r for r in ret['recall']]),
                                  '\t'.join(['%.5f' % r for r in ret['precision']]),
                                  '\t'.join(['%.5f' % r for r in ret['ndcg']]),
                                  '\t'.join(['%.5f' % r for r in ret['rmrr']]))
                    print(pretrain_ret)
                    with torch.cuda.device('cuda:' + str(args.gpu_id)):
                        torch.cuda.empty_cache()
                    range_mrr, range_recall, range_pre = range_test(model, model_smgcn,
                                                                list(data_generator.test_users),
                                                                list(data_generator.test_items_hot))
                    range_pretrain_ret = "range pretrained model \t MRR score=%.5f," \
                                         "recall=[%s], precision=[%s]" % (
                        range_mrr,
                        '\t'.join(['%.5f' % r for r in range_recall]),
                        '\t'.join(['%.5f' % r for r in range_pre])
                    )
                    print(range_pretrain_ret)
                    print("end:", save_id)
            else:
                cur_best_pre_0 = 0.
                print('no model, without pretraining.')
        print("end:")
        print(1 / 0)
    else:
        cur_best_pre_0 = 0.
        print('no pretrain, without pretraining.')

    """
    ***************************************************************************************************************************
    Train.
    ***************************************************************************************************************************
    """
    loss_loger, pre_loger, rec_loger, ndcg_loger, rmrr_loger = [], [], [], [], []
    range_mrr_loger, range_f1_loger, range_acc_loger, range_recall_loger, range_pre_loger = [], [], [], [], []
    mf_loss_loger, emb_loss_loger, const_loss_loger = [], [], []
    should_stop = False
    mf_best_idx = 0
    idx_epoch = 0
    # ********************************************************************************************************
    # *******************************************Train*************************************
    # ********************************************************************************************************
    # load smgcn embedding训练后的模型
    layer = '-'.join([str(l) for l in eval(args.layer_size)])
    mess_dr = '-'.join([str(l) for l in eval(str(args.mess_dropout))])
    weights_save_path = '%sweights-CLEPR/%s/%s/%s/%s/%s/' % (
        args.weights_path, args.dataset, model.model_type, layer, args.embed_size,
        args.model_wpath
    )
    if args.use_S1 == 1:     # pretrain S1
        print('load the pretrained model parameters from: ', weights_save_path)
        model = torch.load(weights_save_path + 'model.pkl', map_location=lambda storage, loc: storage).to(args.device)   # 第一阶段的模型
        print("load embedding model is : {}".format(model))
        model.add_module("attention_layer", SelfAttention(model.mlp_predict_weight_size_list[0],
                                                          attn_dropout_prob=args.attn_dropout_prob))
        model = model.to(args.device)
        print("after add attention layer  model is : {}".format(model))
    model_smgcn = torch.load(weights_save_path + 'model.pkl', map_location=lambda storage, loc: storage).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    ret = test(model, list(data_generator.test_users),
               data_generator.test_group_set_repeat, drop_flag=True, repeat=1)  # 测试使用集合列表

    cur_best_pre_0 = round(ret['precision'][0], 5)
    cur_best_pre_0_const = 0.
    print("SMGCN 的best precision is: ", cur_best_pre_0)
    save_idx = 0
    for epoch in range(args.epoch):
        t4 = time()
        loss, mf_loss, emb_loss, reg_loss, contrastive_loss = 0., 0., 0., 0., 0.

        n_batch = data_generator.n_train // args.batch_size + 1

        for idx in range(n_batch):
            optimizer.zero_grad()
            users, user_set, items, item_set = data_generator.sample(args)
            users = torch.tensor(users, dtype=torch.float32).to(args.device)  # [B, n_user]  multi-hot
            user_set = torch.tensor(user_set, dtype=torch.long).to(args.device)  # list:[] 在一个batch中不重复的user id
            items = torch.tensor(items, dtype=torch.float32).to(args.device)  # [B, n_item]  multi-hot
            item_weights = torch.tensor(data_generator.item_weights, dtype=torch.float32).to(
                args.device)  # [n_item, 1] 每个item在数据集中出现的频率权重
            item_padding_set, user_padding_set = data_generator.train_items_padding, data_generator.train_users_padding
            user_embeddings, all_user_embeddins, ia_embeddings, item_embeddings, cor_value = \
                model(users, item_padding_set=item_padding_set, items=items, user_set=user_set,
                      user_padding_set=user_padding_set)
            item_sets_repeat = torch.tensor(data_generator.item_sets_repeat, dtype=torch.float32).to(
                args.device)  # [B, n_item]  multi-hot)   一个batch中，每个症状集合对应的所有中药
            if args.hard_neg == 0 or args.use_S1 == 0:  # no neg sample or CLEPR-S1
                if args.alg_type == 'CLEPR':
                    batch_mf_loss, batch_emb_loss, batch_reg_loss, batch_contrastive_loss = \
                        model.create_set2set_loss(items, item_weights, user_embeddings, all_user_embeddins,
                                                  ia_embeddings,
                                                  item_embeddings, use_const=1,
                                                  items_repeat=item_sets_repeat, repeat=0,
                                                  # !!!!!! 1: S1 0: smgcn and S2
                                                  logits_user_neg=torch.zeros([0, 1]).to(args.device),
                                                  cor_value=cor_value)
                else:  # CLEPR(S1) and SMGCN
                    batch_contrastive_loss = 0
                    batch_mf_loss, batch_emb_loss, batch_reg_loss, _ = \
                        model.create_set2set_loss(items, item_weights, user_embeddings, all_user_embeddins,
                                                  ia_embeddings, item_embeddings,
                                                  use_const=0,
                                                  items_repeat=item_sets_repeat,
                                                  repeat=1)  # !!! repeat---1: CLEPR(S1) 0: SMGCN
            else:  # add neg sample
                # 得到 top k （step = 6）
                user_embeddings_smgcn, _, ia_embeddings_smgcn, _, _ = \
                    model_smgcn(users, items=items, user_set=user_set)
                logits_user_neg = model_smgcn.get_hard_sample(user_embeddings_smgcn, ia_embeddings_smgcn,
                                                              model_two_stage=model)
                batch_mf_loss, batch_emb_loss, batch_reg_loss, batch_contrastive_loss = \
                    model.create_set2set_loss(items, item_weights, user_embeddings, all_user_embeddins,
                                              ia_embeddings,
                                              item_embeddings, use_const=1,
                                              items_repeat=item_sets_repeat, repeat=0,  # !!!!!! 1: S1 0: smgcn and S2
                                              logits_user_neg=logits_user_neg, cor_value=cor_value)
            batch_loss = batch_mf_loss + batch_emb_loss + batch_reg_loss + batch_contrastive_loss
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss.item()
            mf_loss += batch_mf_loss.item()
            emb_loss += batch_emb_loss.item()
            reg_loss += batch_reg_loss.item()
            contrastive_loss += batch_contrastive_loss.item()
        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()
        loss_loger.append(loss)
        mf_loss_loger.append(mf_loss)
        emb_loss_loger.append(emb_loss)
        const_loss_loger.append(contrastive_loss)
        # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
        if (epoch + 1) % 10 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f + %.5f  ]' % (
                    epoch, time() - t4, loss, mf_loss, emb_loss, reg_loss, contrastive_loss)
                print(perf_str)
            continue

        t5 = time()

        group_to_test = data_generator.test_group_set
        # ret = test(model, list(data_generator.test_users), group_to_test, drop_flag=True)
        ret = test(model, list(data_generator.test_users),
                   data_generator.test_group_set_repeat, drop_flag=True, repeat=1)  # 测试使用集合列表
        range_mrr, range_recall, range_pre = range_test(model, model_smgcn,
                                                    list(data_generator.test_users), list(data_generator.test_items_hot))

        range_f1, range_acc = 0., 0.
        t6 = time()

        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        rmrr_loger.append(ret['rmrr'])
        range_mrr_loger.append(range_mrr)
        range_f1_loger.append(range_f1)
        range_acc_loger.append(range_acc)
        range_recall_loger.append(range_recall)
        range_pre_loger.append(range_pre)

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f +  %.5f ]\n recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f],  ndcg=[%.5f, %.5f], RMRR=[%.5f, %.5f]\n' \
                       'range: acc_score=%.5f, f1 score=%.5f, MRR score=%.5f, ' \
                       'range_recall=[%.5f, %.5f], range_pre=[%.5f, %.5f]' % \
                       (epoch, t5 - t4, t6 - t5, loss, mf_loss, emb_loss, reg_loss, contrastive_loss, ret['recall'][0],
                        ret['recall'][-1],
                        ret['precision'][0], ret['precision'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1], ret['rmrr'][0], ret['rmrr'][-1],
                        range_acc, range_f1, range_mrr,
                        range_recall[0], range_recall[-1],
                        range_pre[0], range_pre[-1])
            print(perf_str)
            paras = str(args.lr) + "_" + str(args.regs) + "_" + str(args.mess_dropout) + "_" + str(
                args.embed_size) + "_" + str(args.adj_type) + "_" + str(args.alg_type)
            print("paras\t", paras)
        cur_best_pre_0_const, stopping_step, should_stop = no_early_stopping(ret['precision'][0], cur_best_pre_0_const,
                                                                             stopping_step, expected_order='acc')

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            print('early stopping')
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if args.use_S1 == 0:
            if (ret['precision'][0] > cur_best_pre_0_const) and args.save_flag == 1:
                weights_save_path1 = weights_save_path + 'l%s_r%s_messdr%s/' % (
                    str(args.lr), '-'.join([str(r) for r in eval(args.regs)]), mess_dr,
                )
                print("\n", "*" * 80, "model sava path",
                      weights_save_path1 + '/model.pkl')
                ensureDir(weights_save_path1 + '/')
                cur_best_pre_0_const = ret['precision'][0]
                torch.save(model, weights_save_path1 + args.two_save_tail + '/' + str(save_idx) + '/model.pkl')
                print('save the weights in path: ',
                      weights_save_path1 + args.two_save_tail + '/' + str(save_idx) + '/model.pkl')
                mf_best_idx = idx_epoch
        else:
            if (ret['precision'][0] >= cur_best_pre_0) or ((ret['precision'][0] > cur_best_pre_0_const)) and args.save_flag == 1:
                weights_save_path1 = weights_save_path + 'l%s_r%s_messdr%s_t%s_cor_%s_step%s_max_step_len_%s/' % (
                    str(args.lr), '-'.join([str(r) for r in eval(args.regs)]), mess_dr,
                    str(args.t),
                    str(args.co_lamda),
                    str(args.step),
                    str(args.max_step_len)
                )
                print("\n", "*" * 80, "model sava path",
                      weights_save_path1 + args.two_save_tail + '/' + str(save_idx) + '/model.pkl')
                ensureDir(weights_save_path1 + args.two_save_tail + '/' + str(save_idx) + '/')
                cur_best_pre_0_const = ret['precision'][0]
                torch.save(model, weights_save_path1 + args.two_save_tail + '/' + str(save_idx) + '/model.pkl')
                print('save the weights in path: ',
                      weights_save_path1 + args.two_save_tail + '/' + str(save_idx) + '/model.pkl')
                mf_best_idx = idx_epoch
        idx_epoch += 1

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    rmrrs = np.array(rmrr_loger)
    range_mrrs = np.array(range_mrr_loger)
    range_f1s = np.array(range_f1_loger)
    range_accs = np.array(range_acc_loger)
    range_recalls = np.array(range_recall_loger)
    range_pres = np.array(range_pre_loger)
    idx = mf_best_idx
    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s],  ndcg=[%s], rmrr=[%s]\n" \
                 "range: acc score=%.5f, f1 score=%.5f, MRR score=%.5f, " \
                 "range_recall=[%s], range_pre=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]),
                  '\t'.join(['%.5f' % r for r in rmrrs[idx]]),
                  range_accs[idx],
                  range_f1s[idx],
                  range_mrrs[idx],
                  '\t'.join(['%.5f' % r for r in range_recalls[idx]]),
                  '\t'.join(['%.5f' % r for r in range_pres[idx]]))
    print(final_perf)

    save_path = '%soutput/%s/%s.result-SMGCN-%d' % (args.proj_path, args.dataset, model.model_type, args.result_index)
    ensureDir(save_path)
    f = open(save_path, 'a')

    cur_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    f.write(
        '\ntime=%s, fusion=%s, embed_size=%d, lr=%s, layer_size=%s, mess_dropout=%s, regs=%s, adj_type=%s\n\t%s\t'
        % (str(cur_time), args.fusion, args.embed_size, str(args.lr), args.layer_size,
           args.mess_dropout, args.regs,
           args.adj_type, final_perf))
    f.close()
    endTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('end ', endTime)

    print('loger load to tensorboard success')
