import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# 1. Scaled Dot-Product Attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    """

               queries: (batch_size, num_queries, d)
               keys: (batch_size, num_kv_pairs, d)
               values: (batch_size, num_kv_pairs, d_v)
               为什么queries和keys的长度相等都为d：
               因为我们需要计算每个查询（query）和每个键（key）之间的相关性或兼容性，而计算两个向量的相关性（通常使用点积）要求它们的维度相同。
               valid_lens: (batch_size,) or (batch_size, num_queries)
               如何理解queries或者是keys的batch_size：
               1.机器翻译时：假设一次处理4个句子（batch_size=4）
               2.图像分类任务：假设处理16张图片（batch_size=16）
           Returns:
               output: (batch_size, num_queries, d_v)
               attention_weights: (batch_size, num_queries, num_kv_pairs)
           """
    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        #当用多头注意力机制和Scaled Dot-Product Attention相结合时：
        # Q: (batch, heads, q_len, d_k)
        # K^T: (batch, heads, d_k, k_len)
        # scores: (batch, heads, q_len, k_len)

        # attn_weights: (batch, heads, q_len, k_len)
        # V: (batch, heads, k_len, d_v)
        # output: (batch, heads, q_len, d_v)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        # 计算注意力分数: Q * K^T / sqrt(d) （d是每个查询/键向量的维度大小）
        # 利用transpose方法将keys从(batch_size, num_kv_pairs, d)转置为(batch_size, d, num_kv_pairs)
        # (batch_size, num_queries, d) × (batch_size, d, num_kv_pairs)
        # 利用sqrt(d)缩放是点积方差保持在1左右

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, V)

        return output, attn_weights


# 2. Multi-Head Attention，多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        # num_hiddens 表示隐藏层维度，也就是注意力机制的输出维度和内部表示维度。head_dim = num_hiddens // num_heads
        super().__init__()
        assert d_model % num_heads == 0
        #判断d_model是否可以被整除

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        #为什么要先对query，key，value进行线性变换再分头：
        #增加模型的表达能力：线性变换引入了可学习的参数，使得模型可以自适应地调整Query、Key和Value的表示，从而增强模型的表达能力。

        self.attention = ScaledDotProductAttention(dropout)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        Q = self.W_Q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        attn_output, attn_weights = self.attention(Q, K, V, mask)
        # 通过__call__方法自动调用forward函数，nn.module固定调用forward函数
        #每个头独立计算注意力
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        #transpose() 操作可能产生不连续内存布局的张量view() 操作要求张量在内存中是连续的
        # contiguous() 会创建新的内存连续副本（如果需要）

        return self.W_O(attn_output)


# 3. Add & Norm Layer
class AddNorm(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        return self.norm(x + self.dropout(sublayer_output))
    #为什么不把Dropout放在残差相加之前：因为如果Dropout放在外面，可能会随机丢弃原始输入X的信息，破坏了残差连接保护原始信息的设计初衷
    #Dropout(X + Y)   不推荐

# 4. Feed Forward Network
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# 5. Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.add_norm1 = AddNorm(d_model, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.add_norm1(x, attn_output)

        ff_output = self.feed_forward(x)
        x = self.add_norm2(x, ff_output)

        return x


# 6. Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers=6, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


# 测试代码
def test_transformer():
    # 创建随机输入
    batch_size, seq_len, d_model = 2, 10, 512
    x = torch.randn(batch_size, seq_len, d_model)

    # 创建Transformer Encoder
    encoder = TransformerEncoder(num_layers=3, d_model=d_model, num_heads=8)

    # 前向传播
    output = encoder(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")


    return encoder, output


if __name__ == "__main__":

    model, output = test_transformer()
