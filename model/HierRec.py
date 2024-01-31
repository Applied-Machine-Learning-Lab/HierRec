from utils.layers import *


class HierRec(torch.nn.Module):
    # mlp_dims=[[64,64],64, 64, [64,64]]
    def __init__(self, field_dims, embed_dim, mlp_dims, im_num, dropout, device):
        super().__init__()
        self.emb_dim = embed_dim
        self.im_num = im_num
        self.mlp_dims = mlp_dims
        self.len_field_dims = len(field_dims)
        self.emb_alldim = embed_dim*(self.len_field_dims-1)
        self.domain_num = field_dims[0]
        self.embedding = FeaturesEmbedding(field_dims[1:], embed_dim)
        self.dnn1 = MultiLayerPerceptron(self.emb_alldim, mlp_dims[0], dropout=dropout)
        self.explicit = explicit_gen(self.domain_num, self.emb_dim, self.len_field_dims,
                                         inout_dim=[mlp_dims[0][-1], mlp_dims[1][0], mlp_dims[1][1]], im_num=im_num,
                                         dropout=dropout)
        self.implicit = implicit_gen(im_num, embed_dim, self.len_field_dims,
                                         inout_dim=[mlp_dims[1][1], mlp_dims[2][0], mlp_dims[2][1]],
                                         dropout=dropout)
        self.out_trans = MultiLayerPerceptron(im_num*mlp_dims[2][1], [mlp_dims[2][1]], dropout=dropout)
        self.dnn3 = MultiLayerPerceptron(mlp_dims[2][1], mlp_dims[3], dropout=dropout, output_layer=True)

    def forward(self, x):
        domain = x[:, 0]
        x = x[:, 1:]
        # emb_x = N*(len-1)*emb_dim
        emb_x = self.embedding(x)
        # ex_net = N*inout_dim[0]*inout_dim[1]
        # impara = N*im_num*(len-1)*1
        ex_net_1, ex_net_2, ex_bias_1, ex_bias_2, im_para = self.explicit(domain)
        # im_net = N*im_num*inout_dim[0]*inout_dim[1]
        im_net_1, im_net_2, im_bias_1, im_bias_2 = self.implicit(emb_x, im_para)

        # x = N*mlp_dims[0]
        x = self.dnn1(emb_x.view(-1, self.emb_alldim))
        # N*1*out = mat(N*1*in  --   N*inout_dim[0]*inout_dim[1])+(N*1*out)
        x = torch.matmul(x.unsqueeze(1), ex_net_1) + ex_bias_1
        x = torch.matmul(x, ex_net_2) + ex_bias_2
        # N*im_num*out = (mat(N*1*1*in  --  N*im_num*inout_dim[0]*inout_dim[1])+N*im_num*1*out).squeeze()
        x = (torch.matmul(x.unsqueeze(1), im_net_1) + im_bias_1)
        x = (torch.matmul(x, im_net_2) + im_bias_2).squeeze().view(-1, self.im_num*self.mlp_dims[2][1])
        x = self.out_trans(x)
        x = torch.sigmoid(self.dnn3(x).squeeze())
        return x
