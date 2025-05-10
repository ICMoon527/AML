from turtle import forward
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import math


class SVM(nn.Module):
    def __init__(self, args, input_dim, output_dim) -> None:
        super(SVM, self).__init__()
        self.args = args

        self.model = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        y = self.model(x)
        return y

class DNN(nn.Module):
    def __init__(self, args, input_dim, output_dim) -> None:
        super(DNN, self).__init__()
        self.args = args

        self.layers = []
        self.activate = {
            'elu': nn.ELU,
            'relu': nn.ReLU,
            'softplus': nn.Softplus
        }[args.nonlin]

        times = 2
        self.layers.append(self.fullConnectedLayer(input_dim, output_dim*192*times, args.batchnorm))
        self.layers.append(self.fullConnectedLayer(output_dim*192*times, output_dim*48*times, args.batchnorm))
        self.layers.append(self.fullConnectedLayer(output_dim*48*times, output_dim*12*times, args.batchnorm))
        # self.layers.append(nn.Linear(input_dim/4, output_dim*256))
        self.layers.append(nn.Linear(output_dim*12*times, output_dim))

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        y = self.model(x)
        return y

    def fullConnectedLayer(self, input_dim, output_dim, batchnorm=False):
        if batchnorm:
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                self.activate(),
                nn.Dropout(p=self.args.dropout_rate, inplace=False)  # 激活函数之后
            )
        else:
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                self.activate(),
                nn.Dropout(p=self.args.dropout_rate, inplace=False)  # 激活函数之后
            )


class ATTDNN(nn.Module):
    def __init__(self, args, input_dim, output_dim) -> None:
        super(ATTDNN, self).__init__()
        self.args = args

        self.layers = []
        self.activate = {
            'elu': nn.ELU,
            'relu': nn.ReLU,
            'softplus': nn.Softplus
        }[args.nonlin]

        self.layers.append(AttentionLayer(args, input_dim, args.nClasses*input_dim, args.nClasses))
        # self.layers.append(self.fullConnectedLayer(input_dim, 1024, args.batchnorm))
        # self.layers.append(self.fullConnectedLayer(1024, 2048, args.batchnorm))
        # self.layers.append(self.fullConnectedLayer(2048, 256, args.batchnorm))
        # self.layers.append(nn.Linear(256, output_dim))
        self.layers.append(self.fullConnectedLayer(input_dim, output_dim, args.batchnorm))

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        y = self.model(x)
        return y

    def fullConnectedLayer(self, input_dim, output_dim, batchnorm=False):
        if batchnorm:
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                self.activate()
            )
        else:
            return nn.Sequential(
                nn.Dropout(p=self.args.dropout_rate, inplace=False),
                nn.Linear(input_dim, output_dim),
                self.activate()
            )


class AttentionLayer(nn.Module):
    def __init__(self, args, in_size, hidden_size, num_attention_heads):
        """
        Softmax(Q@K.T)@V
        """
        super(AttentionLayer, self).__init__()

        self.activate = {
            'elu': nn.ELU,
            'relu': nn.ReLU,
            'softplus': nn.Softplus,
            'sigmoid': nn.Sigmoid
        }[args.nonlin]

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)

        self.key_layer = nn.Linear(in_size, hidden_size)
        self.query_layer = nn.Linear(in_size, hidden_size)
        self.value_layer = nn.Linear(in_size, hidden_size)

    def forward(self, x):
        key = self.activate()(self.key_layer(x))  # (batch, hidden_size)
        query = self.activate()(self.query_layer(x))
        value = self.activate()(self.value_layer(x))

        key_heads = self.trans_to_multiple_heads(key)  # (batch, heads_num, head_size)
        query_heads = self.trans_to_multiple_heads(query)
        value_heads = self.trans_to_multiple_heads(value)

        attention_scores = torch.matmul(key_heads, query_heads.permute(0, 2, 1))  # (batch, heads_num, heads_num)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores_normalized = F.softmax(attention_scores, dim=-1)

        out = torch.matmul(attention_scores_normalized, value_heads)  # (batch, heads_num, head_size)

        return out.mean(dim=-2)


    def trans_to_multiple_heads(self, x):
        new_size = x.size()[ : -1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_size)
        return x


class UDNN(nn.Module):
    def __init__(self, args, input_dim, output_dim, hidden_dim=2048) -> None:
        super(UDNN, self).__init__()
        self.args = args
        self.activate = {
            'relu': nn.ReLU,
            'elu': nn.ELU,
            'softplus': nn.Softplus
        }[args.nonlin]

        self.initial_layer = self.fullConnectedLayer(input_dim, args.initial_dim) # expand dim
        
        self.down_1 = self.fullConnectedLayer(hidden_dim, hidden_dim/2)
        self.down_2 = self.fullConnectedLayer(hidden_dim/2, hidden_dim/4)
        self.down_3 = self.fullConnectedLayer(hidden_dim/4, hidden_dim/8)

        self.up_1 = self.fullConnectedLayer(hidden_dim/8, hidden_dim/4)
        self.up_2 = self.fullConnectedLayer(hidden_dim/4, hidden_dim/2)
        self.up_3 = self.fullConnectedLayer(hidden_dim/2, hidden_dim)
        
        self.output_layer = self.fullConnectedLayer(args.initial_dim, output_dim)

    def forward(self, x):

        # initial_out = self.initial_layer(x)  # 2048
        # down_1_out = self.down_1(initial_out)  # 1024
        # down_2_out = self.down_2(down_1_out)  # 512

        # down_3_out = self.down_3(down_2_out)  # 256
        
        # up_1_out = self.up_1(down_3_out)  # 512
        # up_2_out = self.up_2((up_1_out+down_2_out)/2)  # 1024
        # up_3_out = self.up_3((up_2_out+down_1_out)/2)  # 2048
        # out = self.output_layer((up_3_out+initial_out)/2)

        initial_out = self.initial_layer(x)  # 256
        up_1_out = self.up_1(initial_out)  # 512
        up_2_out = self.up_2(up_1_out)  # 1024
        up_3_out = self.up_3(up_2_out)  # 2048

        down_1_out = self.down_1(up_3_out)  # 1024
        down_2_out = self.down_2(down_1_out)  # 512
        down_3_out = self.down_3(down_2_out)  # 256
        out = self.output_layer(down_3_out)

        return out

    def fullConnectedLayer(self, input_dim, output_dim):

        return nn.Sequential(
            nn.Dropout(p=self.args.dropout_rate, inplace=False),
            nn.Linear(int(input_dim), int(output_dim)),
            self.activate()
        )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        assert (
            self.head_dim * num_heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.num_heads different pieces
        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        queries = query.reshape(N, query_len, self.num_heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Scaled dot-product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) / (self.head_dim ** 0.5)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy, dim=-1)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.embed_size
        )

        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        feature_dim,  # 输入特征维度
        embed_size,   # 嵌入维度
        num_layers,
        num_heads,
        device,
        forward_expansion,
        dropout,
        max_length,
        chunk_size=128  # 每个块的最大长度
    ):
        super(TransformerEncoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        
        # 如果输入特征维度与嵌入维度不同，则添加线性投影层
        if feature_dim != embed_size:
            self.projection = nn.Linear(feature_dim, embed_size)
        else:
            self.projection = nn.Identity()  # 如果相同，则直接传递

        self.position_embedding = PositionalEncoding(embed_size, max_length)

        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, num_heads, dropout, forward_expansion) for _ in range(num_layers)]
        )

        self.dropout = nn.Dropout(dropout)
        self.chunk_size = chunk_size
        self.attention_pooling = FeatureAttentionPooling(embed_size)

    def forward(self, x, mask=None):
        N, seq_length, feature_dim = x.shape
        out = self.projection(x)  # 应用线性投影
        # out = self.position_embedding(out)

        # 分块处理
        chunks = []
        for i in range(0, seq_length, self.chunk_size):
            chunk = out[:, i:i+self.chunk_size]
            chunk_mask = mask[:, :, i:i+self.chunk_size, i:i+self.chunk_size] if mask is not None else None
            
            # 通过每一层 Transformer Block
            for layer in self.layers:
                chunk = layer(chunk, chunk, chunk, chunk_mask)
            
            chunks.append(chunk)

        # 将所有块重新组合成一个张量
        out = torch.cat(chunks, dim=1)

        # pooled_output = self.attention_pooling(out)

        return out


class Classifier(nn.Module):
    def __init__(self, seq_length, embed_size, num_classes=2):
        super(Classifier, self).__init__()
        self.seq_length = seq_length
        self.embed_size = embed_size
        self.flatten = nn.Flatten(start_dim=1)  # 展平除了 batch 维度以外的所有维度
        # self.fc = nn.Linear(seq_length * embed_size, 512)  # 第一个全连接层
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(embed_size * seq_length, num_classes)  # 分类层

    def forward(self, x):
        x = self.flatten(x)
        # x = self.fc(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        x = self.classifier(x)
        return x


# 定义完整的模型
class FullModel(nn.Module):
    def __init__(
        self,
        feature_dim,
        embed_size,
        num_layers,
        num_heads,
        device,
        forward_expansion,
        dropout,
        max_length,
        seq_length,
        num_classes=2,
        chunk_size=128
    ):
        super(FullModel, self).__init__()
        self.encoder = TransformerEncoder(
            feature_dim,
            embed_size,
            num_layers,
            num_heads,
            device,
            forward_expansion,
            dropout,
            max_length,
            chunk_size
        )
        self.classifier = Classifier(seq_length, embed_size, num_classes)

    def forward(self, x, mask=None):
        x = self.encoder(x, mask)
        x = self.classifier(x)
        return x


def create_padding_mask(seq, pad_token=0):
    mask = (seq != pad_token).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
    return mask.to(dtype=torch.float32)

class AttentionPooling(nn.Module):
    def __init__(self, embed_size):
        super(AttentionPooling, self).__init__()
        self.embed_size = embed_size
        self.query = nn.Parameter(torch.randn(embed_size))  # 可学习的查询向量

    def forward(self, x):
        # x: (batch_size, seq_length, embed_size)
        # 计算注意力分数
        attention_scores = torch.matmul(x, self.query) / (self.embed_size ** 0.5)  # (batch_size, seq_length)
        attention_weights = torch.softmax(attention_scores, dim=-1)  # (batch_size, seq_length)
        # 加权求和
        pooled_output = torch.sum(x * attention_weights.unsqueeze(-1), dim=1)  # (batch_size, embed_size)
        return pooled_output
    
class FeatureAttentionPooling(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        # 可学习的特征级注意力参数
        self.attention = nn.Sequential(
            nn.Linear(embed_size, 1),  # 每个特征维度输出1个分数
        )
    
    def forward(self, x):
        """
        输入: (batch_size, seq_length, embed_size)
        输出: (batch_size, seq_length)
        """
        # 计算特征注意力权重 (batch_size, seq_length, embed_size) -> (batch_size, seq_length, 1)
        pooled = self.attention(x)
        return pooled


if __name__ == '__main__':
    # 参数设置
    feature_dim = 2  # 输入特征维度
    embed_size = 256  # 模型内部使用的嵌入维度
    num_layers = 6
    num_heads = 8  # 注意力头数必须能够整除 embed_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    forward_expansion = 4
    dropout = 0.1
    max_length = 9  # 输入序列的最大长度
    seq_length = 9  # 输入序列的实际长度
    chunk_size = 9  # 因为输入序列较短，这里设为等于序列长度
    num_classes = 2  # 二分类任务

    # 初始化完整模型
    model = FullModel(
        feature_dim,
        embed_size,
        num_layers,
        num_heads,
        device,
        forward_expansion,
        dropout,
        max_length,
        seq_length,
        num_classes,
        chunk_size
    ).to(device)

    # 测试数据
    x = torch.randn(2, seq_length, feature_dim).to(device)  # Example input tensor with shape (batch_size, seq_length, feature_dim)
    mask = create_padding_mask(x.sum(dim=-1))  # 创建掩码，假设非零元素为有效

    # 调用模型
    out = model(x, mask)
    print(out.shape)  # 应该输出 [batch_size, num_classes]

    # 打印输出以检查是否符合预期
    print(out)