# -*- coding: utf-8 -*-
# @Time    : 2022/1/10 10:40
# @Author  :
# @File    : parser.py
# @Description :
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run SMGCN.")
    parser.add_argument('--weights_path', nargs='?', default='',
                        help='Store model path.')
    parser.add_argument('--data_path', nargs='?', default='./data/',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='',
                        help='Project path.')
    parser.add_argument('--result_index', type=int, default=1,
                        help='result file index.')
    parser.add_argument('--test_file', nargs='?', default='test.txt',
                        help='valid_5percent.txt')
    parser.add_argument('--train_file', nargs='?', default='train_id.txt',
                        help='train_id.txt')
    parser.add_argument('--result_label', nargs='?', default='',
                        help='result path label.')
    parser.add_argument('--is_test_data', type=int, default=0,
                        help='flag for test data.')
    parser.add_argument('--dataset', nargs='?', default='Herb',
                        help='Choose a dataset from {Herb, NetEase, gowalla, yelp2018, amazon-book}')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=2000,
                        help='Number of epoch.')
    parser.add_argument('--embed_size', type=int, default=32,
                        help='Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[64,128]',
                        help='Output sizes of every layer')
    parser.add_argument('--pair_layer_size', nargs='?', default='[32]',
                        help='Output sizes of every layer')
    parser.add_argument('--mlp_layer_size', nargs='?', default='[128]',
                        help='Output sizes of every layer')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size.')
    parser.add_argument('--fusion', nargs='?', default='add',
                        help='fusion method.')
    parser.add_argument('--regs', nargs='?', default='[7e-3]',
                        help='embed Regularizations.')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate.')
    parser.add_argument('--model_type', nargs='?', default='ngcf',
                        help='Specify the name of model (ngcf).')
    parser.add_argument('--adj_type', nargs='?', default='norm',
                        help='Specify the type of the adjacency (laplacian) matrix from {plain, norm, mean}.')
    parser.add_argument('--alg_type', nargs='?', default='CLEPR',
                        help='Specify the type of the graph convolutional layer from {CLEPR, SMGCN, ngcf, gcn, gcmc}.')
    parser.add_argument('--gpu_id', type=int, default=2,
                        help='0 for NAIS_prod, 1 for NAIS_concat')
    parser.add_argument('--node_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.0,0.0]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--Ks', nargs='?', default='[5,20]',
                        help='Output sizes of every layer')
    parser.add_argument('--save_flag', type=int, default=0,
                        help='0: Disable model saver, 1: Activate model saver')
    parser.add_argument('--loss_weight', type=float, default=1.0,
                        help='number:0-1 change different loss')
    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')
    parser.add_argument('--loss_w', type=float, default=1,
                        help='change different loss weight')
    parser.add_argument('--save_tail', nargs='?', default='',
                        help='the model save path tail.')
    parser.add_argument('--contrast_reg', type=float, default=1.0,
                        help='reduce contrast loss')
    parser.add_argument('--t', type=float, default=1.0,
                        help='temperature parameter')
    parser.add_argument('--model_wpath', nargs='?', default='date_2022-09-18_60_ori_emb_seed1234',
                        help='load model path for pretrain')
    parser.add_argument('--attn_dropout_prob', type=float, default=0.0,
                        help='attention dropout radio.')
    parser.add_argument('--two_save_tail', nargs='?', default='',
                        help='the model save path tail for two stage.')
    parser.add_argument('--all_dataset', type=int, default=0,
                        help='loss items use batch or all train dataset repeat set')
    parser.add_argument('--set_K', nargs='?', default='[5, 10, 15, 20]',
                        help='set recommendation size')
    parser.add_argument('--seed', type=int, default=1234,
                        help='seed for random')
    parser.add_argument('--step', type=int, default=6,
                        help='neg sample generate step')
    parser.add_argument('--max_step_len', type=int, default=26,
                        help='neg sample generate max step len')
    parser.add_argument('--attention', type=int, default=0,
                        help="use self attention for item set embedding")
    parser.add_argument('--co_lamda', type=float, default=0.0,
                        help='self correlation add loss to contrastive loss.')
    parser.add_argument('--hard_neg', type=int, default=1,
                        help='using neg sample or not')
    parser.add_argument('--use_S1', type=int, default=1,
                        help='use the first stage or not')
    return parser.parse_args()
