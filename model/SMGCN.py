# -*- coding: utf-8 -*-
# @Time    : 2022/1/10 10:18
# @Author  :
# @File    : CLEPR.py
# @Description :  the pytorch version of CLEPR

import os
import sys
from utils.helper import *
from utils.batch_test import *
# from utils.batch_test_case_study import *
import datetime
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from model.Attention_layer import SelfAttention
import math

class SMGCN(nn.Module):
    def __init__(self, data_config, pretrain_data):
        super(SMGCN, self).__init__()
        self.model_type = 'CLEPR'
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type

        self.pretrain_data = pretrain_data
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.n_fold = 100
        self.norm_adj = data_config['norm_adj']
        self.sym_pair_adj = data_config['sym_pair_adj']
        self.herb_pair_adj = data_config['herb_pair_adj']
        self.n_nonzero_elems = self.norm_adj.count_nonzero()
        self.lr = args.lr
        # self.link_lr = args.link_lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.loss_weight = args.loss_weight

        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)
        self.device = args.device


        self.fusion = args.fusion
        print('***********fusion method************ ', self.fusion)

        self.mlp_predict_weight_size = eval(args.mlp_layer_size)
        self.mlp_predict_n_layers = len(self.mlp_predict_weight_size)
        print('mlp predict weight ', self.mlp_predict_weight_size)
        print('mlp_predict layer ', self.mlp_predict_n_layers)
        self.model_type += '_%s_%s_l%d' % (self.adj_type, self.alg_type, self.n_layers)

        self.regs = eval(args.regs)
        print('regs ', self.regs)
        self.decay = self.regs[0]
        self.verbose = args.verbose

        '''
        *********************************************************
        Create embedding for Input Data & Dropout.
        '''

        self.mess_dropout = args.mess_dropout


        """
        *********************************************************
        Create Model Parameters (i.e., Initialize Weights)
        """
        # initialization of model parameters
        self.weights = self._init_weights()

        """
        *********************************************************
        self-attention for item embedding and item set embedding
        """
        if args.attention == 1:
            self.attention_layer = SelfAttention(self.mlp_predict_weight_size_list[0],
                                                 attn_dropout_prob=args.attn_dropout_prob)

    # 初始化权重，存在all weight字典中，键为权重的名字，值为权重的值
    def _init_weights(self):
        # xavier init
        initializer = nn.init.xavier_uniform_
        all_weights = nn.ParameterDict()
        all_weights.update({'user_embedding': nn.Parameter(initializer(torch.empty(self.n_users, self.emb_dim)))})
        all_weights.update({'item_embedding': nn.Parameter(initializer(torch.empty(self.n_items, self.emb_dim)))})
        if self.pretrain_data is None:
            print('using xavier initialization')
        else:
            # pretrain
            all_weights['user_embedding'].data = self.pretrain_data['user_embed']
            all_weights['item_embedding'].data = self.pretrain_data['item_embed']
            print('using pretrained initialization')

        self.weight_size_list = [self.emb_dim] + self.weight_size    # [embedding size(64), layer_size(128, 256)]
        pair_dimension = self.weight_size_list[len(self.weight_size_list) - 1]
        for k in range(self.n_layers):
            w_gc_user = torch.empty([2 * self.weight_size_list[k], self.weight_size_list[k + 1]])
            b_gc_user = torch.empty([1, self.weight_size_list[k + 1]])
            W_gc_item = torch.empty([2 * self.weight_size_list[k], self.weight_size_list[k + 1]])
            b_gc_item = torch.empty([1, self.weight_size_list[k + 1]])
            Q_user = torch.empty([self.weight_size_list[k], self.weight_size_list[k]])
            Q_item = torch.empty([self.weight_size_list[k], self.weight_size_list[k]])
            all_weights.update({'W_gc_user_%d' % k: nn.Parameter(initializer(w_gc_user))})    # w,b 第K层聚合user信息的权重矩阵
            all_weights.update({'b_gc_user_%d' % k: nn.Parameter(initializer(b_gc_user))})
            all_weights.update({'W_gc_item_%d' % k: nn.Parameter(initializer(W_gc_item))})   # w, b 第K层聚合item信息的权重矩阵
            all_weights.update({'b_gc_item_%d' % k: nn.Parameter(initializer(b_gc_item))})
            all_weights.update({'Q_user_%d' % k: nn.Parameter(initializer(Q_user))})      # 第K层构建user邻居信息时的权重矩阵
            all_weights.update({'Q_item_%d' % k: nn.Parameter(initializer(Q_item))})    # 第K层构建item邻居信息时的权重矩阵

        self.mlp_predict_weight_size_list = [self.mlp_predict_weight_size[
                                                 len(self.mlp_predict_weight_size) - 1]] + self.mlp_predict_weight_size
        print('mlp_predict_weight_size_list ', self.mlp_predict_weight_size_list)

        for k in range(self.mlp_predict_n_layers):
            W_predict_mlp_user = torch.empty([self.mlp_predict_weight_size_list[k], self.mlp_predict_weight_size_list[k + 1]])
            b_predict_mlp_user = torch.empty([1, self.mlp_predict_weight_size_list[k + 1]])
            all_weights.update({'W_predict_mlp_user_%d' % k: nn.Parameter(initializer(W_predict_mlp_user))})
            all_weights.update({'b_predict_mlp_user_%d' % k: nn.Parameter(initializer(b_predict_mlp_user))})
            # all_weights.update({'W_predict_mlp_item_%d' % k: nn.Parameter(initializer(W_predict_mlp_user))})
            # all_weights.update({'b_predict_mlp_item_%d' % k: nn.Parameter(initializer(b_predict_mlp_user))})
        print("\n", "#" * 75, "pair_dimension is ", pair_dimension)
        M_user = torch.empty([self.emb_dim, pair_dimension])
        M_item = torch.empty([self.emb_dim, pair_dimension])
        all_weights.update({'M_user': nn.Parameter(initializer(M_user))})
        all_weights.update({'M_item': nn.Parameter(initializer(M_item))})
        return all_weights

    # todo: 矩阵分解，加速计算
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.tensor([coo.row, coo.col], dtype=torch.long).to(args.device)
        # v = torch.from_numpy(coo.data).float().to(args.device)
        v = torch.tensor(np.array(coo.data), dtype=torch.float32).to(args.device)
        return torch.sparse.FloatTensor(i, v, coo.shape)

    # todo: 矩阵分解，加速计算
    def _split_A_hat(self, X):
        A_fold_hat = []
        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]).to(args.device))
        return A_fold_hat

    # 使用图卷积神经网络得到的user embedding
    def _create_graphsage_user_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)   # 将矩阵按行分为n_hold份，每份矩阵存在列表元素中
        pre_embeddings = torch.cat([self.weights['user_embedding'], self.weights['item_embedding']], 0)  # [n_user+n_item, B]

        # print("*" * 20, "embeddings", pre_embeddings)
        all_embeddings = [pre_embeddings]
        for k in range(self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(torch.sparse.mm(A_fold_hat[f], pre_embeddings))   # 矩阵分解相乘，加速计算
            # 分解的矩阵拼回成一个, 每行表示邻居节点传来的信息
            embeddings = torch.cat(temp_embed, 0)   # 前n_user行表示与该user相关的item邻居传递来的信息和其本身，后n_item行同理
            embeddings = torch.tanh(torch.matmul(embeddings, self.weights['Q_user_%d' % k]))   # 构建邻居信息
            embeddings = torch.cat([pre_embeddings, embeddings], 1)    # 消息聚合，将上一层和目前的表示串联

            pre_embeddings = torch.tanh(
                torch.matmul(embeddings, self.weights['W_gc_user_%d' % k]) + self.weights['b_gc_user_%d' % k])  # 高阶信息传播
            pre_embeddings = nn.Dropout(self.mess_dropout[k])(pre_embeddings)

            norm_embeddings = F.normalize(pre_embeddings, p=2, dim=1)
            all_embeddings = [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, 1)
        # 但由于是使用了user的权重矩阵, 所以这里仅将user的embedding拿出来计算
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], 0)

        temp = torch.sparse.mm(self._convert_sp_mat_to_sp_tensor(self.sym_pair_adj).to(args.device),
                                             self.weights['user_embedding'])      # 利用user-user图计算user的embedding
        user_pair_embeddings = torch.tanh(torch.matmul(temp, self.weights['M_user']))

        if self.fusion in ['add']:
            u_g_embeddings = u_g_embeddings + user_pair_embeddings
        if self.fusion in ['concat']:
            u_g_embeddings = torch.cat([u_g_embeddings, user_pair_embeddings], 1)
        return u_g_embeddings

    def _create_graphsage_item_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)

        pre_embeddings = torch.cat([self.weights['user_embedding'], self.weights['item_embedding']], 0)

        all_embeddings = [pre_embeddings]
        for k in range(self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(torch.sparse.mm(A_fold_hat[f], pre_embeddings))
            embeddings = torch.cat(temp_embed, 0)

            embeddings = torch.tanh(torch.matmul(embeddings, self.weights['Q_item_%d' % k]))
            embeddings = torch.cat([pre_embeddings, embeddings], 1)

            pre_embeddings = torch.tanh(
                torch.matmul(embeddings, self.weights['W_gc_item_%d' % k]) + self.weights['b_gc_item_%d' % k])

            pre_embeddings = nn.Dropout(self.mess_dropout[k])(pre_embeddings)

            norm_embeddings = F.normalize(pre_embeddings, p=2, dim=1)
            all_embeddings = [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], 0)

        temp = torch.sparse.mm(self._convert_sp_mat_to_sp_tensor(self.herb_pair_adj).to(args.device),
                                             self.weights['item_embedding'])
        item_pair_embeddings = torch.tanh(torch.matmul(temp, self.weights['M_item']))

        if self.fusion in ['add']:
            i_g_embeddings = i_g_embeddings + item_pair_embeddings

        if self.fusion in ['concat']:
            i_g_embeddings = torch.cat([i_g_embeddings, item_pair_embeddings], 1)

        return i_g_embeddings

    def create_batch_rating(self, pos_items, user_embeddings):
        pos_scores = torch.sigmoid(torch.matmul(user_embeddings, pos_items.transpose(0, 1)))
        return pos_scores

    def get_self_correlation(self, item_embeddings):
        """
        Args:
            item_embeddings:  [B, max_item_len, emb] 两者相乘，减去对角阵后即为每个药方内部，两个中药的相似度矩阵，将这个相似度矩阵的和作为
            对比学习的一个约束加入loss中，这个loss需要调参，使得两两之间的相似度比较大
        Returns:
            self_correlation_matrix: 自相关矩阵的
        """
        cor_matrix = torch.cosine_similarity(item_embeddings.unsqueeze(2), item_embeddings.unsqueeze(1), dim=-1)   # [B, max_len, max_len]
        diag = torch.diagonal(cor_matrix, dim1=1, dim2=2)
        a_diag = torch.diag_embed(diag)
        cor_matrix = cor_matrix - a_diag   # 对角元素置为0，即自己和自己不去计算相似度
        cor_matrix = cor_matrix.pow(2)     # 平方值得值大于0
        cor_value = torch.sum(cor_matrix) / 2    # 计算整个相似度矩阵的总和，padding的部分是0，所以不会计算到 这个值越小越好
        return cor_value


    def get_set_embedding(self, item_padding_set, ia_embeddings):
        """
        Args:
            item_padding_set: list: [B, max_batch_item_len]  item id set并被padding后 列表id
            ia_embeddings: [n_items, emb]
        过程：
        ia_embeddings -- > [n_item+1, emb] 最后一行是padding的嵌入表示
        Returns: set_embedding [B, emb]
        """
        padding_embedding = torch.zeros((1, ia_embeddings.size(1)), dtype=torch.float32).to(args.device)
        ia_padding_embedding = torch.cat((ia_embeddings, padding_embedding), 0)  # [n_item + 1, emb]
        item_embeddings = ia_padding_embedding[item_padding_set, :]  # [B, max_batch_item_len, emb]
        if item_embeddings.size(0) > 1024 or args.co_lamda == 0.0:  # valid and test
            cor_value = 0
        else:
            cor_value = self.get_self_correlation(item_embeddings)   # value
        position_ids = torch.arange(data_generator.max_item_len, dtype=torch.long, device=args.device).unsqueeze(1)
        d_model = item_embeddings.size(2)
        pe = torch.zeros(data_generator.max_item_len, d_model, device=args.device)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model)).to(args.device)
        pe[:, 0::2] = torch.sin(position_ids * div_term)
        pe[:, 1::2] = torch.cos(position_ids * div_term)
        # pe[:, :] = 1 / (position_ids.T + 1)
        pe = 1 / (position_ids + 1)
        position_embedding = pe.repeat(item_embeddings.shape[0], 1, 1)  # 位置编码 [B, max_batch_item_len, emb]
        attention_mask, value_attention_mask, presci_adj_matrix = self.get_attention_mask(item_embeddings)
        if args.attention == 1:
            item_embeddings, item_attention_scores = self.attention_layer(item_embeddings,
                                                                          attention_mask=attention_mask,
                                                                          value_attention_mask=value_attention_mask,
                                                                          presci_adj_matrix=presci_adj_matrix
                                                                          )  # 经过self attention 层[B, max_batch_item_len, emb]

            neigh = torch.sum(value_attention_mask, dim=2) / value_attention_mask.size(2)  # [B, max_len]
            neigh_num = torch.sum(neigh, dim=1)  # [B, 1]
            item_set_embedding = torch.sum(item_embeddings, dim=1)  # [B, emb]
            normal_matrix_item = torch.reciprocal(neigh_num)
            normal_matrix_item = normal_matrix_item.unsqueeze(1)
            # 复制embedding_size列  [B, embedding_size]
            extend_normal_item_embeddings = normal_matrix_item.repeat(1, item_set_embedding.shape[1])
            # 对应元素相乘
            item_set_embedding = torch.mul(item_set_embedding, extend_normal_item_embeddings)  # 平均池化
            return item_set_embedding, item_attention_scores, item_embeddings, cor_value

    def get_attention_mask(self, item_seq):
        """Generate attention mask for attention."""
        attention_mask = (item_seq != 0).to(dtype=torch.float32)  # [B, max_len, emb]
        # attention_p = attention_mask[0][0]
        item_len = torch.sum(attention_mask, dim=1)   # [B, emb]
        item_len_matrix = item_len[:, :1]  # [B, 1]  得到每个药方的长度
        extended_attention_mask = torch.bmm(attention_mask, attention_mask.permute(0, 2, 1))  # [B, max_len, max_len]
        extended_attention_mask = (extended_attention_mask > 0).to(dtype=torch.float32)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)  # fp16 compatibility\
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        presci_adj_matrix = None
        # attention_mask = (1.0 - attention_mask) * - 10000.0
        return extended_attention_mask, attention_mask, presci_adj_matrix

    def get_hard_sample(self, user_embeddings, ia_embeddings, model_two_stage=None):
        step = args.step
        max_step_len = args.max_step_len
        random_id = [id for id in range(0, max_step_len-step)]
        rating = self.create_batch_rating(ia_embeddings, user_embeddings)
        vals, indices = rating.sort(descending=True)
        logits_user_neg = None
        k_id = 0
        for k in random_id:
            # topK_items = indices[:, k:]
            topK_items = indices[:, k:k+step]
            if args.attention == 1:
                # padding = torch.tensor([data_generator.n_items] * (k)).to(args.device)
                padding = torch.tensor([data_generator.n_items] * (data_generator.max_item_len-k)).to(args.device)
                padding = padding.unsqueeze(1)
                padding = padding.repeat(1, topK_items.size(0)).transpose(0, 1)
                topK_items = torch.cat([topK_items, padding], dim=1).cpu().numpy().tolist()
                if model_two_stage is None:
                    topK_sets, _, topK_att_embedding, _ = self.get_set_embedding(topK_items, ia_embeddings)  # [B, emb]
                else:
                    topK_sets, _, topK_att_embedding, _ = model_two_stage.get_set_embedding(topK_items, ia_embeddings)  # [B, emb]
            else:
                topK_items = topK_items.cpu().numpy().tolist()
                item_set_embeddings = ia_embeddings[topK_items, :]  # [B, k,emb]
                topK_sets = torch.sum(item_set_embeddings, dim=1) / step  # 平均池化
                # for ks in range(0, self.mlp_predict_n_layers):
                #     topK_sets = F.relu(
                #         torch.matmul(topK_sets, self.weights['W_predict_mlp_item_%d' % ks])
                #         + self.weights['b_predict_mlp_item_%d' % ks])
                #     topK_sets = F.dropout(topK_sets, self.mess_dropout[ks])  # [B, emb] 药方归纳, item set的整体表示
            if k_id == 0:
                logits_user_neg = torch.mul(user_embeddings, topK_sets)  # [B, emb]
                logits_user_neg = torch.sum(logits_user_neg, dim=1).unsqueeze(1)  # [B, 1]  症状集合*对应的topk 集合
            else:
                neg = torch.mul(user_embeddings, topK_sets)
                neg = torch.sum(neg, dim=1).unsqueeze(1)
                logits_user_neg = torch.cat([logits_user_neg, neg], dim=1)  # [B, 12（5-65）] 每一行是每个top k集合的得分
            k_id += 1
        return logits_user_neg

    def create_set2set_loss(self, items, item_weights, user_embeddings, all_user_embeddins,
                            ia_embeddings, item_embeddings, use_const=0, logits_user_neg=None,
                            items_repeat=None, repeat=0, neg_item_embeddings=None, cor_value=0):
        # item_embeddings [B, emd]
        if repeat == 1:
            items = items_repeat
        predict_probs = torch.sigmoid(torch.matmul(user_embeddings, ia_embeddings.transpose(0, 1)))
        mf_loss = torch.sum(torch.matmul(torch.square((items - predict_probs)), item_weights), 0)
        # mf_loss = nn.MSELoss(reduction='elementwise_mean')(items, predict_probs)
        mf_loss = mf_loss / self.batch_size
        all_item_embeddins = ia_embeddings
        regularizer = torch.norm(all_user_embeddins) ** 2 / 2 + torch.norm(all_item_embeddins) ** 2 / 2
        regularizer = regularizer.reshape(1)
        # F.normalize(all_user_embeddins, p=2) + F.normalize(all_item_embeddins, p=2)
        regularizer = regularizer / self.batch_size

        emb_loss = self.decay * regularizer

        reg_loss = torch.tensor([0.0], dtype=torch.float64, requires_grad=True).to(args.device)
        if use_const == 0:
            return mf_loss, emb_loss, reg_loss, reg_loss
        else:
            """
            user_embeddings: [B, n_e]
            item_embeddings: [B, n_e]
            """
            loss_use_cos = args.save_tail

            # 使用固定的t, 对比学习loss使用sup nce
            t = torch.tensor(args.t).to(args.device)
            # Normalize to unit vectors
            user_embeddings = F.normalize(user_embeddings, p=2, dim=1)
            item_embeddings = F.normalize(item_embeddings, p=2, dim=1)
            logits_user_batch = torch.matmul(user_embeddings, item_embeddings.transpose(0, 1)) / t  # [B, B]
            if logits_user_neg is None:
                logits_user_neg = self.get_hard_sample(user_embeddings, ia_embeddings)
                logits_user = torch.cat([logits_user_batch, logits_user_neg], dim=1)  # [B， B+N]
            elif logits_user_neg.size(0) == 0:
                logits_user = logits_user_batch  # [B， B]
            else:
                logits_user = torch.cat([logits_user_batch, logits_user_neg], dim=1)  # [B， B]

            #######################################################################################################
            # 使用SupConLoss：有监督的对比学习loss，可以处理多个正例和负例

            labels = torch.tensor(data_generator.train_positive_list).to(args.device)
            labels = labels.contiguous().view(-1, 1)  # [B, 1] 列向量
            mask = torch.eq(labels, labels.T).float().to(args.device)  # [B, B]
            if logits_user_neg.size(0) != 0:
                mask = torch.cat((mask, torch.zeros_like(logits_user_neg).to(args.device)), dim=1)  # [B, B + len(neg)]
            # for numerical stability
            logits_max, _ = torch.max(logits_user, dim=1, keepdim=True)  # 每行的最大值 [B,1]
            logits = logits_user - logits_max.detach()  # [B, B + len(neg)]
            exp_logits = torch.exp(logits)   # [B, B + len(neg)]
            # exp_logits = exp_logits / torch.exp(logits_sim_score)    # 张凯老师那边处理自适应的权重
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)  # 仅计算正例，分子是正例
            # loss
            loss = -t * mean_log_prob_pos
            loss = loss.view(1, logits_user.shape[0]).mean()
            contrastive_loss = loss
            contrastive_loss = args.contrast_reg * contrastive_loss
            cor_loss = args.co_lamda * cor_value
            if cor_loss > 0:
                reg_loss = cor_loss   # 只是为了打印出来看下值
            contrastive_loss = contrastive_loss + cor_loss
            return mf_loss, emb_loss, reg_loss, contrastive_loss


    def normalize(self, *xs):
        return [None if x is None else F.normalize(x, dim=-1) for x in xs]

    def forward(self, users, item_padding_set=None, items=None, user_set=None, pos_items=None, train=True, user_padding_set=None):
        """
          *********************************************************
          Compute Graph-based Representations of all users & items via Message-Passing Mechanism of Graph Neural Networks.
          Different Convolutional Layers:
              1. ngcf: defined in 'Neural Graph Collaborative Filtering', SIGIR2019;
              2. gcn:  defined in 'Semi-Supervised Classification with Graph Convolutional Networks', ICLR2018;
              3. gcmc: defined in 'Graph Convolutional Matrix Completion', KDD2018;
          """
        # todo: todo 应该在主函数中
        if self.alg_type in ['SMGCN', 'CLEPR']:
            if train:
                ua_embeddings = self._create_graphsage_user_embed()  # [n_user, 最后一层size的大小]: 所有user的embedding
                ia_embeddings = self._create_graphsage_item_embed()  # [n_item, 最后一层size的大小]
                if user_set is None:
                    all_user_embeddins = None
                else:
                    all_user_embeddins = torch.index_select(ua_embeddings, 0,
                                                        user_set)  # [一个Batch中涉及到的user个数, 最后一层embedding_size]: 选择一个batch中user的embedding

                # todo:change: 构建item/user set embedding
                """
                1. 将ia_embedding与multi-hot编码items相乘得到batch中item set的表示 item_set_embeddings
                2. 将item_set_embedding平均池化
                """
                sum_embeddings = torch.matmul(users, ua_embeddings)  # [B,  最后一层embedding_size]
                normal_matrix = torch.reciprocal(torch.sum(users, 1))
                normal_matrix = normal_matrix.unsqueeze(1)  # [B, 1]
                # 复制embedding_size列  [B, embedding_size]
                extend_normal_embeddings = normal_matrix.repeat(1, sum_embeddings.shape[1])
                # 对应元素相乘
                user_embeddings = torch.mul(sum_embeddings, extend_normal_embeddings)  # 平均池化 [B, emb]
                for k in range(0, self.mlp_predict_n_layers):
                    user_embeddings = F.relu(
                        torch.matmul(user_embeddings, self.weights['W_predict_mlp_user_%d' % k])
                        + self.weights['b_predict_mlp_user_%d' % k])
                    user_embeddings = F.dropout(user_embeddings, self.mess_dropout[k])  # 证候归纳, user set的整体表示 [B, emb]

                cor_value = 0
                if args.attention == 1 and item_padding_set is not None:
                    item_embeddings, _, item_att_embedding, cor_value = self.get_set_embedding(item_padding_set,
                                                                ia_embeddings)  # [B,  最后一层embedding_size]
                else:
                    # *********************************** item set  ***********************************
                    item_set_embeddings = torch.matmul(items, ia_embeddings)   # [B,  最后一层embedding_size]
                    normal_matrix_item = torch.reciprocal(torch.sum(items, 1))
                    normal_matrix_item = normal_matrix_item.unsqueeze(1)  # [B, 1]
                    # 复制embedding_size列  [B, embedding_size]
                    extend_normal_item_embeddings = normal_matrix_item.repeat(1, item_set_embeddings.shape[1])
                    # 对应元素相乘
                    item_embeddings = torch.mul(item_set_embeddings, extend_normal_item_embeddings)  # 平均池化
                    # for k in range(0, self.mlp_predict_n_layers):
                    #     item_embeddings = F.relu(
                    #         torch.matmul(item_embeddings, self.weights['W_predict_mlp_item_%d' % k])
                    #         + self.weights['b_predict_mlp_item_%d' % k])
                    #     item_embeddings = F.dropout(item_embeddings, self.mess_dropout[k])      # 药方归纳, item set的整体表示
                return user_embeddings, all_user_embeddins, ia_embeddings, item_embeddings, cor_value
            else:
                ua_embeddings = self._create_graphsage_user_embed()
                ia_embeddings = self._create_graphsage_item_embed()
                pos_items = torch.tensor(pos_items, dtype=torch.long).to(args.device)
                pos_i_g_embeddings = torch.index_select(ia_embeddings, 0, pos_items)   # 根据test中使用的item id选择item的embedding
                sum_embeddings = torch.matmul(users, ua_embeddings)

                normal_matrix = torch.reciprocal(torch.sum(users, 1))

                normal_matrix = normal_matrix.unsqueeze(1)

                extend_normal_embeddings = normal_matrix.repeat(1, sum_embeddings.shape[1])

                user_embeddings = torch.mul(sum_embeddings, extend_normal_embeddings)

                for k in range(0, self.mlp_predict_n_layers):
                    user_embeddings = F.relu(
                        torch.matmul(user_embeddings,
                                     self.weights['W_predict_mlp_user_%d' % k]) + self.weights[
                            'b_predict_mlp_user_%d' % k])
                    user_embeddings = F.dropout(user_embeddings, self.mess_dropout[k])
                return user_embeddings, pos_i_g_embeddings












