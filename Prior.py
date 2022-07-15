import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from AutoEncoder import Encoder_pre
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt
import math




class Encoder_pre(torch.nn.Module):
    def __init__(self,):
        super(Encoder_pre,self).__init__()
        alpha = 0.3
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, 32,kernel_size=5),
            torch.nn.MaxPool3d(2),
            torch.nn.LeakyReLU(alpha)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(32,64,kernel_size=3),
            torch.nn.MaxPool3d(2),
            torch.nn.LeakyReLU(alpha)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(64,128,kernel_size=3),
            torch.nn.LeakyReLU(alpha)
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(128,256,kernel_size=3),
            torch.nn.LeakyReLU(alpha)
        )

    def forward(self, x):
        bt_size = x.size(0)
        x = x.view(-1,1,32,32,32)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0),-1)
        x = x.view(bt_size,-1,x.size(-1))
        return x

class MHDPA(nn.Module):
    """Multi-Head Dot-Product attention as defined in https://arxiv.org/pdf/1806.01822.pdf"""

    def __init__(self, memory_slots, key_size, value_size, n_heads):
        """
        Args:
            memory_slots: the number of entries in the memory
            key_size: the dimensionality of keys and queries (2nd dim in Q, K matrices)
            value_size: the dimensionality of values (2nd dim in V matrix)
            n_heads: number of separate DPA blocks
            extra_input: a boolean flag to indicate if extra input given to memory before self-attention

        """
        super(MHDPA, self).__init__()
        self.memory_slots = memory_slots
        self.key_size = key_size
        self.value_size = value_size
        self.n_heads = n_heads

        self.num_attention_heads = n_heads  # 8
        self.attention_head_size = int(self.key_size / self.num_attention_heads)  # 16  每个注意力头的维度

        self.all_head_size = int(self.num_attention_heads * self.attention_head_size)

        self.memory_size = self.value_size * self.n_heads
        self.projection_size = 2048
        self.qkv_projector = nn.Linear(self.memory_size, self.projection_size)
        self.qkv_layernorm = nn.LayerNorm(self.projection_size)

        self.qkv_size = n_heads * self.key_size

        self.query = nn.Linear(2048, self.all_head_size)
        self.key = nn.Linear(2048, self.all_head_size)
        self.value = nn.Linear(2048, self.all_head_size)

    def transpose_for_scores(self, x):
        # INPUT:  x'shape = [bs, seqlen, hid_size]  假设hid_size=128
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # [bs, seqlen, 8, 16]
        x = x.view(*new_x_shape)  #
        return x.permute(0, 2, 1, 3)  # [bs, 8, seqlen, 16]

    def forward(self, q,k,v):

        mixed_query_layer = self.query(q)
        mixed_key_layer = self.key(k)
        mixed_value_layer = self.value(v)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        context_layer = self.qkv_layernorm(context_layer)

        return context_layer


class NormMLP(nn.Module):

    def __init__(self, input_size, output_size):
        super(NormMLP, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, activations):
        return self.layer_norm(self.linear(F.relu(activations)))


class ResidualTransform(nn.Module):

    def __init__(self, n_neighbours, key_size, value_size, n_heads, hidden_dim):
        super(ResidualTransform, self).__init__()
        self.attention = MHDPA(n_neighbours, key_size, value_size, n_heads)
        self.norm_mlp = NormMLP(hidden_dim, hidden_dim)

    def forward(self, q, k, v):
        activations = q + self.attention(q,k,v)
        activations = activations + self.norm_mlp(activations)
        return activations


class PriorModule(nn.Module):
    def __init__(self, query_embed_dim, label_embed_dim, n_neighbours,
                 key_size, value_size, n_heads, num_layers):
        super(PriorModule, self).__init__()
        self.query_embed_dim = query_embed_dim
        self.n_neighbours = n_neighbours
        self.key_size = key_size
        self.value_size = value_size
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.label_embed_dim = label_embed_dim

        self.enc = Encoder_pre()
        self.input_size = self.label_embed_dim + self.query_embed_dim +1
        self.hidden_dim = self.value_size * self.n_heads

        # Reserving one extra label for missing labels

        self.residual_layers = ResidualTransform(
            self.n_neighbours, self.key_size, self.value_size, self.n_heads,
            2048)

    def forward(self,buffer_embeddings, buffer_labels, query, distances):

        query = query.unsqueeze(1)
        label_embeds = self.enc(buffer_labels.float())
        memory = self.residual_layers(query, buffer_embeddings.float(), label_embeds)
        query = query.squeeze(1)
        memory = memory.squeeze(1)
        weighted_memory = torch.cat((query,memory), dim=1)

        return weighted_memory


if __name__ == '__main__':
    decoder = PriorModule(
        query_embed_dim=2048,label_embed_dim=2048,n_neighbours=5,
        key_size=2048,value_size=2048,n_heads=2,num_layers=5
    )

    query_embeds = torch.rand(16,2048)
    buffer_embeds = torch.rand(16,5,2048).long()
    buffer_label = torch.rand(16,5,32,32,32).long()
    distance = torch.rand(16,5)

    out = decoder(buffer_embeds,buffer_label,query_embeds,distance)
    print(out.shape)
