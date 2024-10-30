import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class CoordMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super(CoordMLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    

class CalculateAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask):
        attention = torch.matmul(Q,torch.transpose(K, -1, -2))
        # print(attention.shape)
        attention = attention.masked_fill_(mask, -1e9)
        attention = torch.softmax(attention / sqrt(Q.size(-1)), dim=-1)
        attention = torch.matmul(attention,V)
        return attention
    
class CrossAttention(nn.Module):
    def __init__(self,hidden_size,all_head_size,head_num):
        super().__init__()
        self.hidden_size    = hidden_size       # input dim
        self.all_head_size  = all_head_size     # output dim
        self.num_heads      = head_num         
        self.h_size         = all_head_size // head_num

        assert all_head_size % head_num == 0

        # W_Q,W_K,W_V (hidden_size,all_head_size)
        # self.linear_c = nn.Linear(rep_size, hidden_size, bias=False)
        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_output = nn.Linear(all_head_size, hidden_size)

        # normalization
        self.norm = sqrt(all_head_size)

    def print(self):
        print(self.hidden_size,self.all_head_size)
        print(self.linear_k,self.linear_q,self.linear_v)

    def forward(self,x,y,attention_mask):
        batch_size = x.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        # q_s: [batch_size, num_heads, seq_length, h_size]
        q_s = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # k_s: [batch_size, num_heads, seq_length, h_size]
        k_s = self.linear_k(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # v_s: [batch_size, num_heads, seq_length, h_size]
        v_s = self.linear_v(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        attention_mask = attention_mask.eq(0)

        attention = CalculateAttention()(q_s,k_s,v_s,attention_mask)
        # attention : [batch_size , seq_length , num_heads * h_size]
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)
        
        # output : [batch_size , seq_length , hidden_size]
        output = self.linear_output(attention)

        return output


class ReducePoolings(nn.Module):
    def __init__(self, reduction, in_features, out_features):
        super(ReducePoolings, self).__init__()
        self.reduction = reduction
        self.layer = nn.Linear(in_features, out_features)

    def forward(self, input_rep, aa_res_ids, input_aa_len, input_seq_len):
        # input_rep: [batch_size, input_aa_len, feature_dim]
        batch_size, _, feature_dim = input_rep.shape
        
        # init output tensor
        output = torch.zeros((batch_size, max(input_seq_len), feature_dim), device=input_rep.device)
        
        for b in range(batch_size):
            filled_mask = torch.zeros(input_seq_len[b], dtype=torch.bool, device=input_rep.device)
            batch_input_rep = input_rep[b, :input_aa_len[b], :]
            
            for i in range(input_seq_len[b]):
                mask = (aa_res_ids[b, :input_aa_len[b]] == i)
                if mask.sum() > 0:
                    if self.reduction == "max":
                        output[b, i, :], _ = batch_input_rep[mask].max(dim=0)
                    elif self.reduction == "mean":
                        output[b, i, :] = batch_input_rep[mask].mean(dim=0)
                    else:
                        raise ValueError("Unsupported reduction method: {}".format(self.reduction))
                    filled_mask[i] = True 

            # fill the rest of the sequence with the average value
            avg_value =output[b, :input_seq_len[b], :][filled_mask].mean(dim=0) if filled_mask.sum() > 0 else torch.zeros(feature_dim, device=input_rep.device)
            output[b, :input_seq_len[b], :][~filled_mask] = avg_value

        return self.layer(output)