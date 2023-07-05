# from https://github.com/renjie-liang/TSGVZoo/blob/main/models/SeqPANLib/layers.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BaseLib.layers import mask_logits

class Conv1D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True, activation=None):
        super(Conv1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, padding=padding,
                                stride=stride, bias=bias)
        self.activation = activation

    def forward(self, x):
        # suppose all the input with shape (batch_size, seq_len, dim)
        x = x.transpose(1, 2)  # (batch_size, dim, seq_len)
        x = self.conv1d(x)
        return x.transpose(1, 2) if self.activation is None else self.activation(x).transpose(1, 2)# (batch_size, seq_len, dim)

class WordEmbedding(nn.Module):
    def __init__(self, num_words, word_dim, droprate, word_vectors=None):
        super(WordEmbedding, self).__init__()
        self.is_pretrained = False if word_vectors is None else True
        if self.is_pretrained:
            self.pad_vec = nn.Parameter(torch.zeros(size=(1, word_dim), dtype=torch.float32), requires_grad=False)
            unk_vec = torch.empty(size=(1, word_dim), requires_grad=True, dtype=torch.float32)
            nn.init.xavier_uniform_(unk_vec)
            self.unk_vec = nn.Parameter(unk_vec, requires_grad=True)
            self.glove_vec = nn.Parameter(torch.tensor(word_vectors, dtype=torch.float32), requires_grad=False)
        else:
            self.word_emb = nn.Embedding(num_words, word_dim, padding_idx=0)
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, word_ids):
        if self.is_pretrained:
            word_emb = F.embedding(word_ids, torch.cat([self.pad_vec, self.unk_vec, self.glove_vec], dim=0),
                                   padding_idx=0)
        else:
            word_emb = self.word_emb(word_ids)
        return self.dropout(word_emb)


class CharacterEmbedding(nn.Module):
    def __init__(self, num_chars, char_dim, droprate):
        super(CharacterEmbedding, self).__init__()
        self.char_emb = nn.Embedding(num_chars, char_dim, padding_idx=0)
        kernels, channels = [1, 2, 3, 4], [10, 20, 30, 40]
        self.char_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=char_dim, out_channels=channel, kernel_size=(1, kernel), stride=(1, 1), padding=0,
                          bias=True),
                nn.ReLU()
            ) for kernel, channel in zip(kernels, channels)
        ])
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, char_ids):
        char_emb = self.char_emb(char_ids)  # (batch_size, w_seq_len, c_seq_len, char_dim)
        char_emb = self.dropout(char_emb)
        char_emb = char_emb.permute(0, 3, 1, 2)  # (batch_size, char_dim, w_seq_len, c_seq_len)
        char_outputs = []
        for conv_layer in self.char_convs:
            output = conv_layer(char_emb)
            output, _ = torch.max(output, dim=3, keepdim=False)  # reduce max (batch_size, channel, w_seq_len)
            char_outputs.append(output)
        char_output = torch.cat(char_outputs, dim=1)  # (batch_size, sum(channels), w_seq_len)
        return char_output.permute(0, 2, 1)  # (batch_size, w_seq_len, sum(channels))


class Embedding(nn.Module):
    def __init__(self, num_words, num_chars, word_dim, char_dim, droprate, out_dim, word_vectors=None):
        super(Embedding, self).__init__()
        self.word_emb = WordEmbedding(num_words, word_dim, droprate, word_vectors=word_vectors)
        self.char_emb = CharacterEmbedding(num_chars, char_dim, droprate)
        # output linear layer
        self.query_conv1d = Conv1D(in_dim=word_dim+char_dim, out_dim=out_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.q_layer_norm = nn.LayerNorm(out_dim, eps=1e-6)

    def forward(self, word_ids, char_ids):
        word_emb = self.word_emb(word_ids)  # (batch_size, w_seq_len, word_dim)
        char_emb = self.char_emb(char_ids)  # (batch_size, w_seq_len, 100)
        emb = torch.cat([word_emb, char_emb], dim=2)  # (batch_size, w_seq_len, word_dim + 100)
        emb = self.query_conv1d(emb)  # (batch_size, w_seq_len, dim)
        emb = self.q_layer_norm(emb)
        return emb


class PositionalEmbedding(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, num_embeddings, embedding_dim):
        super(PositionalEmbedding, self).__init__()
        self.position_embeddings = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, inputs):
        bsz, seq_length = inputs.shape[:2]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=inputs.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)
        position_embeddings = self.position_embeddings(position_ids)
        return position_embeddings


class VisualProjection(nn.Module):
    def __init__(self, visual_dim, dim, droprate=0.0):
        super(VisualProjection, self).__init__()
        self.drop = nn.Dropout(p=droprate)
        self.video_conv1d = Conv1D(in_dim=visual_dim, out_dim=dim, kernel_size=1, stride=1, bias=True, padding=0)
        # self.video_conv1d = nn.Linear(visual_dim, dim)
        self.v_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        
    def forward(self, visual_features):
        # the input visual feature with shape (batch_size, seq_len, visual_dim)
        visual_features = self.drop(visual_features)
        output = self.video_conv1d(visual_features)  # (batch_size, seq_len, dim)
        output = self.v_layer_norm(output)
        return output


class DepthwiseSeparableConvBlock(nn.Module):
    def __init__(self, dim, kernel_size, droprate, num_layers=4):
        super(DepthwiseSeparableConvBlock, self).__init__()
        self.depthwise_separable_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, groups=dim,
                          padding=kernel_size // 2, bias=False),
                nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, bias=True),
                nn.ReLU(),
            ) for _ in range(num_layers)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(dim, eps=1e-6) for _ in range(num_layers)])
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, x):
        output = x  # (batch_size, seq_len, dim)
        for idx, conv_layer in enumerate(self.depthwise_separable_conv):
            residual = output
            output = self.layer_norms[idx](output)  # (batch_size, seq_len, dim)
            output = output.transpose(1, 2)  # (batch_size, dim, seq_len)
            output = conv_layer(output)
            output = self.dropout(output)
            output = output.transpose(1, 2) + residual  # (batch_size, seq_len, dim)
        return output



class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                groups=in_channels, bias=bias, padding=1)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 
                                kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class DepthwiseSeparableConvBlock2(nn.Module):
    def __init__(self, dim, kernel_size, droprate, num_layers=4):
        super(DepthwiseSeparableConvBlock2, self).__init__()
        self.depthwise_separable_conv = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim, eps=1e-6),
                SeparableConv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, bias=True),
                nn.ReLU(),
            ) for _ in range(num_layers)])
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, x):
        output = x.unsqueeze(2)  # (batch_size, seq_len, dim)
        for conv_layer in self.depthwise_separable_conv:
            residual = output
            output = conv_layer(output)
            output = self.dropout(output) + residual
        return output.squeeze(2)

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, droprate):
        super(MultiHeadAttentionBlock, self).__init__()
        assert dim % num_heads == 0, 'The channels (%d) is not a multiple of attention heads (%d)' % (dim, num_heads)
        self.head_size, self.num_heads, self.dim = int(dim / num_heads), num_heads, dim
        self.dropout = nn.Dropout(p=droprate)
        self.query = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.key = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.value = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.out_layer = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, w_seq_len, head_size)

    @staticmethod
    def combine_last_two_dim(x):
        old_shape = list(x.size())
        new_shape = old_shape[:-2] + [old_shape[-2] * old_shape[-1]]
        return x.reshape(shape=new_shape)

    def forward(self, x, mask=None):
        output = self.layer_norm1(x)  # (batch_size, seq_len, dim)
        output = self.dropout(output)
        # multi-head attention layer
        query = self.transpose_for_scores(self.query(output))  # (batch_size, num_heads, seq_len, head_size)
        key = self.transpose_for_scores(self.key(output))
        value = self.transpose_for_scores(self.value(output))
        attention_scores = torch.matmul(query, key.transpose(-1, -2))  # (batch_size, num_heads, seq_len, seq_len)
        attention_scores = attention_scores / math.sqrt(self.head_size)
        if mask is not None:  # masking
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            attention_scores = mask_logits(attention_scores, mask)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # (batch_size, num_heads, seq_len, seq_len)
        attention_probs = self.dropout(attention_probs)
        value = torch.matmul(attention_probs, value)  # (batch_size, num_heads, seq_len, head_size)
        value = self.combine_last_two_dim(value.permute(0, 2, 1, 3))  # (batch_size, seq_len, dim)
        # intermediate layer
        output = self.dropout(value)
        residual = output + x
        output = self.layer_norm2(residual)
        output = self.dropout(output)
        output = self.out_layer(output)
        output = self.dropout(output) + residual
        return output



def create_attention_mask(from_mask, to_mask, broadcast_ones=False):
    batch_size, from_seq_len = from_mask.shape[:2]
    # _, to_seq_len =to_mask.shape[:2]
    to_mask = to_mask.unsqueeze(1).float()
    if broadcast_ones:
        mask = torch.ones(shape=[batch_size, from_seq_len, 1], dtype=torch.float32)
    else:
        mask = from_mask.unsqueeze(2).float()
    mask = torch.matmul(mask, to_mask)  # (batch_size, from_seq_len, to_seq_len)
    return mask

class BiLinear(nn.Module):
    def __init__(self, configs, in_dim, out_dim, kernel_size, stride, padding, bias):
        super(BiLinear, self).__init__()
        self.dense_1 = Conv1D(in_dim=in_dim, out_dim=out_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.dense_2 = Conv1D(in_dim=in_dim, out_dim=out_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        
        self.bias = bias
        if self.bias:
            b = torch.zeros(out_dim, dtype=torch.float32)
            self.bias_value = nn.Parameter(b)

    def forward(self, input1, input2):
        input1 = self.dense_1(input1)
        input2 = self.dense_1(input2)
        output = input1 + input2
        if self.bias:
            output +=  self.bias_value
        return output
    

class DualAttentionBlock(nn.Module):
    def __init__(self, configs, dim, num_heads, droprate, use_bias, activation=None):
        super(DualAttentionBlock, self).__init__()
        self.dropout = nn.Dropout(p=droprate)

        self.layer_norm_1 = nn.LayerNorm(dim, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(dim, eps=1e-6)
        self.layer_norm_t = nn.LayerNorm(dim, eps=1e-6)

        self.dense_1 =  Conv1D(in_dim=dim, out_dim=dim, bias=use_bias)
        self.dense_2 =  Conv1D(in_dim=dim, out_dim=dim, bias=use_bias)
        self.dense_1_activ = activation

        self.dual_multihead_attention = DualMultiAttention(configs, dim, num_heads, droprate)

    def forward(self, from_tensor, to_tensor,  from_mask, to_mask):
        outputs = self.layer_norm_1(from_tensor)
        to_tensor = self.layer_norm_t(to_tensor)
        outputs = self.dropout(outputs)
        outputs = self.dual_multihead_attention(from_tensor=outputs, to_tensor=to_tensor, 
                                                from_mask=from_mask, to_mask=to_mask)

        outputs = self.dense_1(outputs)
        if self.dense_1_activ:
            outputs = self.dense_1_activ(outputs)
        residual = self.dropout(outputs) + from_tensor
        # feed forward layer
        outputs = self.layer_norm_2(residual)
        outputs = self.dropout(outputs) 
        outputs = self.dense_2(outputs)
        outputs = self.dropout(outputs) + residual
        return outputs


class DualMultiAttention(nn.Module):
    def __init__(self, configs, dim, num_heads, droprate):
        super(DualMultiAttention, self).__init__()
        assert dim % num_heads == 0, 'The channels (%d) is not a multiple of attention heads (%d)' % (dim, num_heads)
        self.droprate = droprate
        self.head_size, self.num_heads, self.dim = int(dim / num_heads), num_heads, dim
        self.dropout = nn.Dropout(p=droprate)
        self.softmax = nn.Softmax(dim=-1)


        self.query = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.f_key = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.f_value = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.t_key = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.t_value = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        
        self.s_dense = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.x_dense = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.s_gate = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.x_gate = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)

        self.guided_dense = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.bilinear_1 = BiLinear(configs=configs, in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.bilinear_2 = BiLinear(configs=configs, in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)

        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.out_layer = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, w_seq_len, head_size)


    def forward(self, from_tensor, to_tensor, from_mask, to_mask):
        B, L, D = from_tensor.shape

        query = self.transpose_for_scores(self.query(from_tensor))  # (batch_size, num_heads, seq_len, head_size)
        f_key = self.transpose_for_scores(self.f_key(from_tensor))
        f_value = self.transpose_for_scores(self.f_value(from_tensor))

        t_key = self.transpose_for_scores(self.t_key(to_tensor))
        t_value = self.transpose_for_scores(self.t_value(to_tensor))

        s_attn_mask = create_attention_mask(from_mask, from_mask, broadcast_ones=False).unsqueeze(1)
        x_attn_mask = create_attention_mask(from_mask, to_mask, broadcast_ones=False).unsqueeze(1)

        s_attn_value = torch.matmul(query, f_key.transpose(-1, -2)) / math.sqrt(float(self.head_size))
        s_attn_value += (1.0 - s_attn_mask) * -1e30
        s_attn_score = self.softmax(s_attn_value)
        s_attn_score = self.dropout(s_attn_score)

        x_attn_value = torch.matmul(query, t_key.transpose(-1, -2)) / math.sqrt(float(self.head_size))
        x_attn_value += (1.0 - x_attn_mask) * -1e30
        x_attn_score = self.softmax(x_attn_value)
        x_attn_score = self.dropout(x_attn_score)

        # compute self-attn value
        s_value = torch.matmul(s_attn_score, f_value).permute([0, 2, 1, 3])
        s_value = torch.reshape(s_value, shape=[B, L, D])
        s_value = self.s_dense(s_value)
        

        # compute cross-attn value
        x_value = torch.matmul(x_attn_score, t_value).permute([0, 2, 1, 3])
        x_value = torch.reshape(x_value, shape=[B, L, D])
        x_value = self.x_dense(x_value)

        s_score = self.s_gate(s_value)
        x_score = self.x_gate(x_value)

        # cross gating
        outputs =s_score * x_value + x_score * s_value
        outputs = self.guided_dense(outputs)

        scores = self.bilinear_1(from_tensor, outputs)
        values = self.bilinear_2(from_tensor, outputs)

        outputs = torch.sigmoid(mask_logits(scores, from_mask.unsqueeze(2))) * values
        return outputs






class FeatureEncoder(nn.Module):
    def __init__(self, dim, max_pos_len, kernel_size=7, num_layers=4, droprate=0.0):
        super(FeatureEncoder, self).__init__()
        self.pos_embedding = PositionalEmbedding(num_embeddings=max_pos_len, embedding_dim=dim)
        self.conv_block = DepthwiseSeparableConvBlock(dim=dim, kernel_size=kernel_size, droprate=droprate,
                                                      num_layers=num_layers)
                                                      

    def forward(self, x):
        features = x + self.pos_embedding(x)  # (batch_size, seq_len, dim)
        features = self.conv_block(features)  # (batch_size, seq_len, dim)
        return features


class CQAttention(nn.Module):
    def __init__(self, dim, droprate=0.0):
        super(CQAttention, self).__init__()
        w4C = torch.empty(dim, 1)
        w4Q = torch.empty(dim, 1)
        w4mlu = torch.empty(1, 1, dim)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C, requires_grad=True)
        self.w4Q = nn.Parameter(w4Q, requires_grad=True)
        self.w4mlu = nn.Parameter(w4mlu, requires_grad=True)
        self.dropout = nn.Dropout(p=droprate)
        self.cqa_linear = Conv1D(in_dim=4 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, context, query, c_mask, q_mask):
        score = self.trilinear_attention(context, query)  # (batch_size, c_seq_len, q_seq_len)
        score_ = nn.Softmax(dim=2)(mask_logits(score, q_mask.unsqueeze(1)))  # (batch_size, c_seq_len, q_seq_len)
        score_t = nn.Softmax(dim=1)(mask_logits(score, c_mask.unsqueeze(2)))  # (batch_size, c_seq_len, q_seq_len)
        score_t = score_t.transpose(1, 2)  # (batch_size, q_seq_len, c_seq_len)
        c2q = torch.matmul(score_, query)  # (batch_size, c_seq_len, dim)
        q2c = torch.matmul(torch.matmul(score_, score_t), context)  # (batch_size, c_seq_len, dim)
        output = torch.cat([context, c2q, torch.mul(context, c2q), torch.mul(context, q2c)], dim=2)
        output = self.cqa_linear(output)  # (batch_size, c_seq_len, dim)
        return output

    def trilinear_attention(self, context, query):
        batch_size, c_seq_len, dim = context.shape
        batch_size, q_seq_len, dim = query.shape
        context = self.dropout(context)
        query = self.dropout(query)
        subres0 = torch.matmul(context, self.w4C).expand([-1, -1, q_seq_len])  # (batch_size, c_seq_len, q_seq_len)
        subres1 = torch.matmul(query, self.w4Q).transpose(1, 2).expand([-1, c_seq_len, -1])
        subres2 = torch.matmul(context * self.w4mlu, query.transpose(1, 2))
        res = subres0 + subres1 + subres2  # (batch_size, c_seq_len, q_seq_len)
        return res


class WeightedPool(nn.Module):
    def __init__(self, dim):
        super(WeightedPool, self).__init__()
        weight = torch.empty(dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, x, mask):
        alpha = torch.tensordot(x, self.weight, dims=1)  # shape = (batch_size, seq_length, 1)
        alpha = mask_logits(alpha, mask=mask.unsqueeze(2))
        alphas = nn.Softmax(dim=1)(alpha)
        pooled_x = torch.matmul(x.transpose(1, 2), alphas)  # (batch_size, dim, 1)
        pooled_x = pooled_x.squeeze(2)
        return pooled_x


class CQConcatenate(nn.Module):
    def __init__(self, dim):
        super(CQConcatenate, self).__init__()
        self.weighted_pool = WeightedPool(dim=dim)
        self.conv1d = Conv1D(in_dim=2 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, context, query, q_mask):
        pooled_query = self.weighted_pool(query, q_mask)  # (batch_size, dim)
        _, c_seq_len, _ = context.shape
        pooled_query = pooled_query.unsqueeze(1).repeat(1, c_seq_len, 1)  # (batch_size, c_seq_len, dim)
        output = torch.cat([context, pooled_query], dim=2)  # (batch_size, c_seq_len, 2*dim)
        output = self.conv1d(output)
        return output


class HighLightLayer(nn.Module):
    def __init__(self, dim):
        super(HighLightLayer, self).__init__()
        self.conv1d = Conv1D(in_dim=dim, out_dim=1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x, mask):
        # compute logits
        logits = self.conv1d(x)
        logits = logits.squeeze(2)
        logits = mask_logits(logits, mask)
        # compute score
        scores = nn.Sigmoid()(logits)
        return scores

    @staticmethod
    def compute_loss(scores, labels, mask, epsilon=1e-12):
        labels = labels.type(torch.float32)
        weights = torch.where(labels == 0.0, labels + 1.0, 2.0 * labels)
        loss_per_location = nn.BCELoss(reduction='none')(scores, labels)
        loss_per_location = loss_per_location * weights
        mask = mask.type(torch.float32)
        loss = torch.sum(loss_per_location * mask) / (torch.sum(mask) + epsilon)
        return loss


class DynamicRNN(nn.Module):
    def __init__(self, dim):
        super(DynamicRNN, self).__init__()
        self.lstm = nn.LSTM(input_size=dim, hidden_size=dim, num_layers=1, bias=True, batch_first=True,
                            bidirectional=False)

    def forward(self, x, mask):
        out, _ = self.lstm(x)  # (bsz, seq_len, dim)
        mask = mask.type(torch.float32)
        mask = mask.unsqueeze(2)
        out = out * mask
        return out


class ConditionedPredictor(nn.Module):
    def __init__(self, dim, num_heads, max_pos_len, droprate=0.0, predictor='rnn'):
        super(ConditionedPredictor, self).__init__()
        self.predictor = predictor
        if predictor == 'rnn':
            self.start_encoder = DynamicRNN(dim=dim)
            self.end_encoder = DynamicRNN(dim=dim)
        else:
            self.encoder = FeatureEncoder(dim=dim, num_heads=num_heads, kernel_size=7, num_layers=4,
                                          max_pos_len=max_pos_len, droprate=droprate)
            self.start_layer_norm = nn.LayerNorm(dim, eps=1e-6)
            self.end_layer_norm = nn.LayerNorm(dim, eps=1e-6)

        self.start_block = nn.Sequential(
            Conv1D(in_dim=2 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            Conv1D(in_dim=dim, out_dim=1, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.end_block = nn.Sequential(
            Conv1D(in_dim=2 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            Conv1D(in_dim=dim, out_dim=1, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x, mask):
        if self.predictor == 'rnn':
            start_features = self.start_encoder(x, mask)  # (batch_size, seq_len, dim)
            end_features = self.end_encoder(start_features, mask)
        else:
            start_features = self.encoder(x, mask)
            end_features = self.encoder(start_features, mask)
            start_features = self.start_layer_norm(start_features)
            end_features = self.end_layer_norm(end_features)
        start_features = self.start_block(torch.cat([start_features, x], dim=2))  # (batch_size, seq_len, 1)
        end_features = self.end_block(torch.cat([end_features, x], dim=2))
        start_logits = mask_logits(start_features.squeeze(2), mask=mask)
        end_logits = mask_logits(end_features.squeeze(2), mask=mask)
        return start_logits, end_logits

    @staticmethod
    def extract_index(start_logits, end_logits):
        start_prob = nn.Softmax(dim=1)(start_logits)
        end_prob = nn.Softmax(dim=1)(end_logits)
        outer = torch.matmul(start_prob.unsqueeze(dim=2), end_prob.unsqueeze(dim=1))
        outer = torch.triu(outer, diagonal=0)
        _, start_index = torch.max(torch.max(outer, dim=2)[0], dim=1)  # (batch_size, )
        _, end_index = torch.max(torch.max(outer, dim=1)[0], dim=1)  # (batch_size, )
        return start_index, end_index

    @staticmethod
    def compute_cross_entropy_loss(start_logits, end_logits, start_labels, end_labels):
        start_loss = nn.CrossEntropyLoss(reduction='mean')(start_logits, start_labels)
        end_loss = nn.CrossEntropyLoss(reduction='mean')(end_logits, end_labels)
        return start_loss + end_loss



class TopSelfAttention2(nn.Module):
    def __init__(self, dim, num_heads, droprate):
        super(TopSelfAttention2, self).__init__()
        self.selfattn = torch.nn.MultiheadAttention(dim, num_heads, dropout=droprate) ### 

    def forward(self, x, mask):
        outputs = self.selfattn(x, x, x, mask.T)
        return outputs[0]

class TopSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, droprate):
        super(TopSelfAttention, self).__init__()
        assert dim % num_heads == 0, 'The channels (%d) is not a multiple of attention heads (%d)' % (dim, num_heads)
        self.head_size, self.num_heads, self.dim = int(dim / num_heads), num_heads, dim
        self.dropout = nn.Dropout(p=droprate)
        self.query = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.key = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.value = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, w_seq_len, head_size)

    def forward(self, inputs, mask):
        B, L, D = inputs.shape
        # reshape & transpose: (batch_size, seq_len, dim) --> (batch_size, num_heads, seq_len, head_size)
        query = self.transpose_for_scores(self.query(inputs))  # (batch_size, num_heads, seq_len, head_size)
        key = self.transpose_for_scores(self.key(inputs))
        value = self.transpose_for_scores(self.value(inputs))
        attention_mask = create_attention_mask(mask, mask, broadcast_ones=False).unsqueeze(1)

        attention_value = torch.matmul(query, key.transpose(-1, -2))  # (batch_size, num_heads, seq_len, seq_len)
        attention_value = attention_value / math.sqrt(self.head_size)
        attention_value += (1.0 - attention_mask) * -1e30


        attention_score = self.softmax(attention_value)
        attention_score = self.dropout(attention_score)
        
        value = torch.matmul(attention_score, value).permute([0, 2, 1, 3])
        value = torch.reshape(value, shape=[B, L, D])
        return value


class FeatureEncoderPredict(nn.Module):
    def __init__(self, dim, num_heads, max_pos_len, droprate, attn_drop):
        super(FeatureEncoderPredict, self).__init__()

        self.pos_embedding = PositionalEmbedding(num_embeddings=max_pos_len, embedding_dim=dim)
        self.conv_block = DepthwiseSeparableConvBlock(dim=dim, kernel_size=7, droprate=droprate, num_layers=4)
        self.layer_norm_1 = nn.LayerNorm(dim)
        self.layer_norm_2 = nn.LayerNorm(dim)
        
        self.dropout = nn.Dropout(droprate)
        self.top_self_attention = TopSelfAttention2(dim=dim, num_heads=num_heads, droprate=attn_drop)
        self.dense = Conv1D(in_dim=dim, out_dim=dim)

    def forward(self, x, mask):
        features = x + self.pos_embedding(x)  # (batch_size, seq_len, dim)
        features = self.conv_block(features)  # (batch_size, seq_len, dim)
        # features = self.attention_block(features, mask=mask)  # (batch_size, seq_len, dim)
        outputs = self.layer_norm_1(features)
        outputs = self.dropout(outputs)
        outputs = self.top_self_attention(outputs, mask)
        residual = self.dropout(outputs) + features
        # feed forward layer
        outputs = self.layer_norm_2(residual)
        outputs = self.dropout(outputs)
        outputs = self.dense(outputs)
        outputs = self.dropout(outputs) + residual
        return outputs


class SeqPANPredictor(nn.Module):
    def __init__(self, configs):
        super(SeqPANPredictor, self).__init__()
        self.feature_encoder = FeatureEncoderPredict(dim=configs.model.dim, num_heads=4, 
                                                max_pos_len= configs.model.max_vlen, 
                                                droprate=configs.model.droprate, attn_drop=configs.model.droprate)


        self.start_layer_norm = nn.LayerNorm(configs.model.dim, eps=1e-6)
        self.end_layer_norm = nn.LayerNorm(configs.model.dim, eps=1e-6)

        self.start_hidden = Conv1D(in_dim=configs.model.dim*2, out_dim=configs.model.dim)
        self.end_hidden = Conv1D(in_dim=configs.model.dim*2, out_dim=configs.model.dim)

        self.start_dense = Conv1D(in_dim=configs.model.dim, out_dim=1)
        self.end_dense = Conv1D(in_dim=configs.model.dim, out_dim=1)

    def forward(self, x, mask):
        start_feat = self.feature_encoder(x, mask)
        end_feat = self.feature_encoder(start_feat, mask)

        start_feat = self.start_layer_norm(start_feat)
        end_feat = self.end_layer_norm(end_feat)

        start_feat = self.start_hidden(torch.concat([start_feat, x], dim=-1))
        end_feat = self.end_hidden(torch.concat([end_feat, x], dim=-1))

        start_logits = self.start_dense(start_feat).squeeze(-1)
        end_logits = self.end_dense(end_feat).squeeze(-1)
        return start_logits, end_logits