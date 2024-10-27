import random
from sru import SRU
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, List
from sklearn.neighbors import NearestNeighbors


class FrameAveraging(nn.Module):
    def __init__(self):
        super(FrameAveraging, self).__init__()
        self.ops = torch.tensor([
            [i, j, k] for i in [-1, 1] for j in [-1, 1] for k in [-1, 1]
        ])

    def create_frame(self, X, mask):
        mask = mask.unsqueeze(-1)
        center = (X * mask).sum(dim=1) / mask.sum(dim=1)
        X = X - center.unsqueeze(1) * mask  # [B, N, 3]
        C = torch.bmm(X.transpose(1, 2), X)  # [B, 3, 3] (Cov)
        L, V = torch.linalg.eigh(C.float().detach(), UPLO='U')  # [B,3,3]
        # _, V = torch.symeig(C.detach(), True)  # [B,3,3]
        F_ops = self.ops.unsqueeze(1).unsqueeze(0).to(X.device) * V.unsqueeze(1)  # [1,8,1,3] x [B,1,3,3] -> [B,8,3,3]
        h = torch.einsum('boij,bpj->bopi', F_ops.transpose(2, 3), X)  # transpose is inverse [B,8,N,3]
        h = h.view(X.size(0) * 8, X.size(1), 3)
        return h, F_ops.detach(), center

    def invert_frame(self, X, mask, F_ops, center):
        X = torch.einsum('boij,bopj->bopi', F_ops, X)
        X = X.mean(dim=1)  # frame averaging
        X = X + center.unsqueeze(1)
        return X * mask.unsqueeze(-1)


class FAEncoder(FrameAveraging):
    def __init__(self, input_size, hidden_dim, n_layers, n_heads, dropout, bidirectional=True, encoder_type='sru'):
        super(FAEncoder, self).__init__()

        self.encoder_type = encoder_type

        if encoder_type == 'sru':
            self.encoder = SRU(
                input_size=input_size,
                hidden_size=hidden_dim // 2,
                projection_size=hidden_dim // 2,
                num_layers=n_layers,
                dropout=dropout,
                bidirectional=bidirectional,
            ).float().cuda() # .cuda() is required...no idea why
        else:
            raise NotImplementedError

        self.encoder.reset_parameters()

    def forward(self, input):
        # X: [B, N, 14, 3]
        S, X, A, h_S = input

        B, N = X.shape[0], X.shape[1]
        mask = X.sum(dim=-1) != 0
        if len(X.shape) == 4:
            X = X[:, :, 0]  # [B, N, 3]

        h, _, _ = self.create_frame(X, mask)  # [B*8, N, 3]
        mask = mask.unsqueeze(1).expand(-1, 8, -1).reshape(B * 8, N)
        # mask = mask.unsqueeze(1).expand(-1, 1, -1).reshape(B * 1, N)
        if h_S is not None:
            h_S = h_S.unsqueeze(1).expand(-1, 8, -1, -1).reshape(B * 8, N, -1)
            # h_S = h_S.unsqueeze(1).expand(-1, 1, -1, -1).reshape(B * 1, N, -1)
            h = torch.cat([h, h_S], dim=-1)

        if self.encoder_type == 'sru':
            h, _ = self.encoder(
                h.transpose(0, 1),
                mask_pad=(~mask.transpose(0, 1).bool())
            )
            h = h.transpose(0, 1)
        elif self.encoder_type == 'transformer':
            h, _ = self.encoder(
                h.float(),
                input_masks=mask.bool(),
            )

        return h.view(B, 8, N, -1).mean(dim=1)  # frame averaging
        # return h.view(B, 1, N, -1).mean(dim=1)  # frame averaging


device = torch.device("cuda")

def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1)).to(device)
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


class E_GCL_RM_Node(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    re
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=False,
                 normalize=False, coords_agg='mean', tanh=False):
        super(E_GCL_RM_Node, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        # coord_mlp = []
        # coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        # coord_mlp.append(act_fn)
        # coord_mlp.append(layer)
        # if self.tanh:
        #     coord_mlp.append(nn.Tanh())
        # self.coord_mlp = nn.Sequential(*coord_mlp)

        self.node_gate = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            nn.ReLU(),
            nn.Linear(hidden_nf, hidden_nf),
            nn.Sigmoid()
        )

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1))

    def edge_model(self, source, target, radial, edge_attr, batch_size, k):
        # if edge_attr is None:  # Unused.
        #     out = torch.cat([source, target, radial], dim=1).to(device)
        # else:
        #     out = torch.cat([source, target, radial, edge_attr], dim=1).to(device)
        # computing m_{ij}
        out = torch.cat([source, target, radial], dim=1).to(device)
        out = self.edge_mlp(out.float())
        # need to use softmax to normalize
        if self.attention:
            attn = self.att_mlp(out)
            att_val = torch.softmax(attn.view(batch_size, -1, k), dim=-1).view(-1, 1)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr, batch_size, k):
        row, col = edge_index
        # row = row.to(device)
        dim = edge_attr.size(-1)
        edge_attr = edge_attr.view(batch_size, -1, k, dim)
        agg = torch.sum(edge_attr, dim=2).view(-1, dim)  # gathered information from K neareast neighbors
        out = x
        if self.residual:
            out = x + self.node_gate(agg) * agg
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat, batch_size, k):
        row, col = edge_index
        # row = row.to(device)
        coord_diff = coord_diff.to(device)
        trans = coord_diff * self.coord_mlp(edge_feat)  # [B * L * 30, 3]
        trans = trans.view(batch_size, -1, k, 3)
        if self.coords_agg == 'sum':
            agg = torch.sum(trans, dim=2).view(-1, 3)
            # agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = torch.mean(trans, dim=2).view(-1, 3)
            # agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1).to(device)  # [KNN]

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None, batch_size=1, k=30):
        """
        h: [batch * length, 320]
        edges: [B * L * L, B * L * L]
        coord: [batch * length, 3]
        """
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)   # [B * L * 30], [B * L * 30, 3]
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr, batch_size, k)   # m_{ij}, [B * L * 30, dim]
        # coord = self.coord_model(coord, edge_index, coord_diff, edge_feat, batch_size, k)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr, batch_size, k)

        return h, coord, edge_attr


class EGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, device='cpu', act_fn=nn.SiLU(), n_layers=4,
                 residual=True, attention=False, normalize=False, tanh=False):
        '''
        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param in_edge_nf: Number of features for the edge features
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.
        '''

        super(EGNN, self).__init__()
        self.input_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(self.input_nf, hidden_nf)

        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL_RM_Node(self.hidden_nf, self.hidden_nf, self.hidden_nf,
                                                        edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual,
                                                        attention=attention, normalize=normalize, tanh=tanh,
                                                        coords_agg="sum"))

        self.to(self.device)

    def forward(self, h, x, edges, edge_attr, k):
        h = self.embedding_in(h)
        h = h.reshape(-1, h.size()[-1])   # [batch * length, hidden]
        x = x.reshape(-1, x.size()[-1])    # [batch * length, 3]
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr, k=k)
        #h = self.embedding_out(h)
        return h, x

def get_surface_aa_feature():
    HYDROPATHY = {"I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8, "W": -0.9, "G": -0.4,
                  "T": -0.7, "S": -0.8, "Y": -1.3, "P": -1.6, "H": -3.2, "N": -3.5, "D": -3.5, "Q": -3.5, "E": -3.5,
                  "K": -3.9, "R": -4.5}  # *
    # VOLUME = {'#': 0, "G": 60.1, "A": 88.6, "S": 89.0, "C": 108.5, "D": 111.1, "P": 112.7, "N": 114.1, "T": 116.1,
    #           "E": 138.4, "V": 140.0, "Q": 143.8, "H": 153.2, "M": 162.9, "I": 166.7, "L": 166.7, "K": 168.6,
    #           "R": 173.4, "F": 189.9, "Y": 193.6, "W": 227.8}
    CHARGE = {**{'R': 1, 'K': 1, 'D': -1, 'E': -1, 'H': 0.1}, **{x: 0 for x in 'ABCFGIJLMNOPQSTUVWXYZ'}} # *
    POLARITY = {**{x: 1 for x in 'RNDQEHKSTY'}, **{x: 0 for x in "ACGILMFPWV"}}
    ACCEPTOR = {**{x: 1 for x in 'DENQHSTY'}, **{x: 0 for x in "RKWACGILMFPV"}}
    DONOR = {**{x: 1 for x in 'RKWNQHSTY'}, **{x: 0 for x in "DEACGILMFPV"}}
    # PMAP = lambda x: [HYDROPATHY[x] / 5, CHARGE[x], POLARITY[x], ACCEPTOR[x], DONOR[x]]
    PMAP = lambda x: [HYDROPATHY[x] / 5, CHARGE[x]]

    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    aa_features = []
    for aa in alphabet:
        aa_features.append(PMAP(aa))
    return torch.Tensor(np.array(aa_features)).to(device)  # [20, 2]


def get_edges(n_nodes, k, indices):
    rows, cols = [], []

    for i in range(n_nodes):
        for j in range(k):
            rows.append(i)
            cols.append(indices[i][j+1])

    edges = [rows, cols]   # L * 30
    return edges


def get_edges_batch(n_nodes, batch_size, coords, k=30):
    rows, cols = [], []
    # batch = torch.tensor(range(batch_size)).reshape(-1, 1).expand(-1, n_nodes).reshape(-1).to(device)
    # edges = knn_graph(coords, k=k, batch=batch, loop=False)
    # edges = edges[[1, 0]]

    for i in range(batch_size):
        # k = min(k, len(coords[i]))
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(coords[i])
        distances, indices = nbrs.kneighbors(coords[i])  # [N, 30]
        edges = get_edges(n_nodes, k, indices)  # [[N*N], [N*N]]
        edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
        rows.append(edges[0] + n_nodes * i)  # every sample in batch has its own graph
        cols.append(edges[1] + n_nodes * i)
    edges = [torch.cat(rows).to(device), torch.cat(cols).to(device)]  # B * L * 30
    return edges

class SurfaceEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = self.build_encoder(args)
        self.encoder_layers = args.encoder_layers
        self.surf_aa_features = get_surface_aa_feature()  # [20, 5]
        self.k = args.knn
        # self.pooling_encoder = FAEncoder(3, args.encoder_embed_dim, 2, 4, 0.1,
        #                                  bidirectional=True, encoder_type='sru')
        self.pooling_encoder = FAEncoder(args.encoder_embed_dim+3, args.encoder_embed_dim, 2, 4, 0.1,
                                         bidirectional=True, encoder_type='sru')

    @classmethod
    def build_encoder(cls, args):
        encoder = EGNN(in_node_nf=2, hidden_nf=args.encoder_embed_dim, out_node_nf=3,
                       in_edge_nf=0, device=device, n_layers=args.decoder_layers, attention=True)
        # encoder = EGNN(in_node_nf=5, hidden_nf=args.encoder_embed_dim, out_node_nf=3,
        #                in_edge_nf=0, device=device, n_layers=args.decoder_layers, attention=True)
        return encoder
        # return FAEncoder(5+3, 128, 2, 4, 0.1, bidirectional=True, encoder_type='sru')

    def forward(
        self,
        coor_features,
        aa_features,
        src_lengths,
    ):
        """
        Run the forward pass for an encoder-decoder model.
        aa_identity_features: [B, L]
        coor_features: [B, L, 4, 3]
        prev_output_tokens: [B, L, N+1]
        attention_score: [B, tgt len, src len]

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        # get surface chemical feature
        bs = coor_features.size(0)
        surf_aa_embs = torch.index_select(self.surf_aa_features, 0, aa_features.reshape(-1).long()).reshape(bs, -1, 2)
        coor_features = coor_features.view(coor_features.size(0), -1, 3)

        # sequence representations
        k = min(self.k, min(src_lengths).item()-1)
        k = self.k
        edges = get_edges_batch(coor_features.size()[1], bs, coor_features.detach().cpu(), k)

        h, coor_features = self.encoder(surf_aa_embs, coor_features, edges, None, k)
        h = h.reshape(bs, -1, h.size()[-1])
        coor_features = coor_features.reshape(bs, -1, 3)
        h = self.pooling_encoder((None, coor_features, None, h))  # [B, L, H]

        # feature = torch.max(h, dim=1)[0]
        # decoder_out = torch.tanh(self.binder_indicator(feature))
        # decoder_out = F.log_softmax(output, dim=-1)  # [batch, 2]
        # encoder_outs["encoder_out"] = [h.transpose(0, 1)]
        return h