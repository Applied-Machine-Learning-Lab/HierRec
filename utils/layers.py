import torch
import torch.nn as nn
import numpy as np


class FeaturesEmbedding(nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, out_softmax=False, output_layer=False, activation=nn.ReLU()):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(nn.Linear(input_dim, embed_dim))
            layers.append(nn.BatchNorm1d(embed_dim))
            layers.append(nn.Dropout(p=dropout))
            layers.append(activation)
            input_dim = embed_dim

        self.mlp = nn.Sequential(*layers)
        if out_softmax:
            self.mlp.add_module('out_softmax', nn.Softmax(dim=1))
        if output_layer:
            self.mlp.add_module('output_l', nn.Linear(input_dim, 1))

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)



class implicit_gen(nn.Module):
    def __init__(self, im_num, embed_dim, len_field_dims, inout_dim, dropout=0.2):
        super().__init__()
        self.im_num = im_num
        self.emb_dim = embed_dim
        self.len_field_dims = len_field_dims
        self.emb_alldim = embed_dim*(len_field_dims-1)
        self.embed_atten = nn.MultiheadAttention(embed_dim, 1, batch_first=True, dropout=dropout)
        self.dnn11 = nn.Linear(self.emb_alldim, inout_dim[0]*inout_dim[1])
        self.dnn12 = nn.Linear(self.emb_alldim, inout_dim[1] * inout_dim[2])
        self.bg11 = nn.Linear(self.emb_alldim, inout_dim[1])
        self.bg12 = nn.Linear(self.emb_alldim, inout_dim[2])
        self.inout_dim = inout_dim
    def forward(self, emb_x, im_para):
        # im_para: N*im_num*(len_field_dims-1)*1
        # N*im_num*(len_field_dims-1)*emb_dim
        emb_x = torch.mul(emb_x.unsqueeze(1), im_para).view(-1, self.len_field_dims-1, self.emb_dim)
        emb_x, emb_weight = self.embed_atten(emb_x, emb_x, emb_x)
        # N*im_num*((len_field_dims-1)*emb_dim)
        emb_x = emb_x.contiguous().view(-1, self.im_num, self.emb_alldim)
        # N*im_num*inout_dim[0]*inout_dim[1]
        net1 = self.dnn11(emb_x).view(-1, self.im_num, self.inout_dim[0], self.inout_dim[1])
        net2 = self.dnn12(emb_x).view(-1, self.im_num, self.inout_dim[1], self.inout_dim[2])
        # N*im_num*1*inout_dim[1]
        net1_bias = self.bg11(emb_x).unsqueeze(2)
        net2_bias = self.bg12(emb_x).unsqueeze(2)
        return net1, net2, net1_bias, net2_bias



class explicit_gen(nn.Module):
    def __init__(self, domain_num, embed_dim, len_field_dims, inout_dim, im_num, dropout=0.2):
        super().__init__()
        self.d_embedding = FeaturesEmbedding([domain_num], embed_dim)
        self.atten = nn.MultiheadAttention(embed_dim,2, dropout=dropout)
        self.dnn11 = nn.Linear(embed_dim, inout_dim[0] * inout_dim[1])
        self.dnn12 = nn.Linear(embed_dim, inout_dim[1] * inout_dim[2])
        self.bg11 = nn.Linear(embed_dim, inout_dim[1])
        self.bg12 = nn.Linear(embed_dim, inout_dim[2])
        if embed_dim % im_num == 0:
            self.atten_part = int(embed_dim / im_num)
            self.dnn2 = nn.ModuleList([nn.Linear(self.atten_part, len_field_dims - 1) for i in range(im_num)])
        else:
            raise ValueError(" 'embed_dim % im_num' must be an interger")
        self.norm_scale = nn.Softmax(dim=-1)
        self.inout_dim = inout_dim
        self.im_num = im_num
        self.embed_dim = embed_dim
        self.domain_num = domain_num
        self.len_field_dims = len_field_dims
    def forward(self, x):
        x = self.d_embedding(x).squeeze()
        # print(weight[:5,:])
        # N*inout_dim[0]*inout_dim[1]
        net1 = self.dnn11(x).view(-1, self.inout_dim[0], self.inout_dim[1])
        net2 = self.dnn12(x).view(-1, self.inout_dim[1], self.inout_dim[2])
        # N*1*inout_dim[1]
        net1_bias = self.bg11(x).unsqueeze(1)
        net2_bias = self.bg12(x).unsqueeze(1)
        # N*im_num*(len_field_dims-1)*1
        para2 = []
        for i in range(self.im_num):
            para2.append(self.dnn2[i](x[:, i * self.atten_part:(i + 1) * self.atten_part]).unsqueeze(1))
        # N*im_num*(len_field_dims-1)*1
        para2 = torch.cat(para2, 1)
        para2 = self.norm_scale(para2).unsqueeze(3)
        return net1, net2, net1_bias, net2_bias, para2


