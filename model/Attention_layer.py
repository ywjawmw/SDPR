# -*- coding: utf-8 -*-
# @Time    : 2022/3/29 16:13
# @Author  : Author
# @File    : Attention_layer.py
# @Description : self-attention layer

import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):
    """
    [multi-head(default=1)] Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (torch.Tensor): the input of the self-attention layer
    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer

    """

    def __init__(self, hidden_size, layer_norm_eps=None, n_heads=1, hidden_dropout_prob=0, attn_dropout_prob=0):
        super(SelfAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        if layer_norm_eps is not None:
            self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        else:
            self.LayerNorm = None

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask=None, value_attention_mask=None, presci_adj_matrix=None):
        """
        Args:
            input_tensor:  [B, max_set_len, hidden_size]
            attention_mask:  [B, max_set_len, max_set_len]

        Returns: hidden_states [B, max_set_len, hidden_size]

        """
        query_layer = self.query(input_tensor)
        key_layer = self.key(input_tensor)
        value_layer = self.value(input_tensor)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # [B, max_len, max_len]

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        if presci_adj_matrix is not None:
            attention_scores = attention_scores + presci_adj_matrix
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = hidden_states * value_attention_mask
        if self.LayerNorm is not None:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states, attention_probs

    def gather_indexes(self, output, gather_index):
        """
        Gathers the vectors at the specific positions over a minibatch
        output: [B max_set_len H]
        """
        # 在expand中的-1表示取当前所在维度的尺寸，也就是表示当前维度不变
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])   # [B 1 H]
        # print("\n gather", gather_index[0])
        # todo:notice 取出每个item_seq的最后一个item的H维向量
        output_tensor = output.gather(dim=1, index=gather_index)        # [B item_num H]
        return output_tensor.squeeze(1)










