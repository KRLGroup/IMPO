import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool, MemPooling, BatchNorm
from torch_scatter import scatter_mean, scatter_max
from torch_geometric.utils import to_dense_adj, dense_to_sparse, subgraph, unbatch_edge_index
import math


def readout(x, batch):
    x_mean = scatter_mean(x, batch, dim=0)
    x_max, _ = scatter_max(x, batch, dim=0) 
    return torch.cat((x_mean, x_max), dim=-1)


"""
"""

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Conv2d, KLDivLoss, Linear, Parameter

from torch_geometric.utils import to_dense_batch, degree

EPS = 1e-15



"""
"""

class IMPO(torch.nn.Module):
    r"""Memory based pooling layer from `"Memory-Based Graph Networks"
    <https://arxiv.org/abs/2002.09518>`_ paper, which learns a coarsened graph
    representation based on soft cluster assignments

    .. math::
        S_{i,j}^{(h)} &= \frac{
        (1+{\| \mathbf{x}_i-\mathbf{k}^{(h)}_j \|}^2 / \tau)^{
        -\frac{1+\tau}{2}}}{
        \sum_{k=1}^K (1 + {\| \mathbf{x}_i-\mathbf{k}^{(h)}_k \|}^2 / \tau)^{
        -\frac{1+\tau}{2}}}

        \mathbf{S} &= \textrm{softmax}(\textrm{Conv2d}
        (\Vert_{h=1}^H \mathbf{S}^{(h)})) \in \mathbb{R}^{N \times K}

        \mathbf{X}^{\prime} &= \mathbf{S}^{\top} \mathbf{X} \mathbf{W} \in
        \mathbb{R}^{K \times F^{\prime}}

    Where :math:`H` denotes the number of heads, and :math:`K` denotes the
    number of clusters.

    Args:
        in_channels (int): Size of each input sample :math:`F`.
        out_channels (int): Size of each output sample :math:`F^{\prime}`.
        heads (int): The number of heads :math:`H`.
        num_clusters (int): number of clusters :math:`K` per head.
        tau (int, optional): The temperature :math:`\tau`. (default: :obj:`1.`)
    """
    def __init__(self, in_channels: int, out_channels: int, heads: int,
                 num_clusters_per_class: int, num_classes: int, tau: float = 1., lambda1=10, lambda2=0.01):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.num_clusters1 = num_clusters_per_class*num_classes
        self.num_clusters2 = num_clusters_per_class*(num_classes+1)
        self.num_clusters_per_class = num_clusters_per_class
        self.num_classes = num_classes
        self.tau = tau
        self.cluster_map = torch.arange(num_classes+1).repeat((num_clusters_per_class, 1)).T.reshape(-1)

        self.k = Parameter(torch.Tensor(heads, self.num_clusters2, in_channels))
        self.conv = Conv2d(heads, 1, kernel_size=1, padding=0, bias=False)
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.reset_parameters()
        self.use_gumbel = False

    def reset_parameters(self):
        torch.nn.init.orthogonal_(self.k)
        
    def impo_loss(self, S: Tensor, S_raw: Tensor, y: Tensor, mask: Tensor) -> Tensor:
        B,N,K = S.shape

        y = y.reshape(-1,1).repeat((1,N))
        assigned_probs, assigned = S.max(-1)
        assigned_class = self.cluster_map[assigned].to(y.device)
        
        
        mask_right = ((assigned_class<self.num_classes)&(assigned_class==y)).long()&mask
        mask_wrong = ((assigned_class<self.num_classes)&(assigned_class!=y)).long()&mask
        ass_loss =   (mask_wrong*assigned_probs).mean() 
        
        sum_loss = S.sum(1)[:,:self.num_clusters1].sum(1).mean()
        
        
        m = mask_right.bool()|mask_wrong.bool()
        m = mask&m
        
        bincount = torch.ones(self.num_classes).to(y.device)
        if m.sum()>0:
            bincount += y[m].bincount()
        
        cel = F.cross_entropy(S[m][:,:-1], y[m], weight=bincount.max().item()/(bincount+1e-3))
        
        l = self.lambda1*cel + self.lambda2*sum_loss
        return l
    @staticmethod
    def kl_loss(S: Tensor) -> Tensor:
        r"""The additional KL divergence-based loss

        .. math::
            P_{i,j} &= \frac{S_{i,j}^2 / \sum_{n=1}^N S_{n,j}}{\sum_{k=1}^K
            S_{i,k}^2 / \sum_{n=1}^N S_{n,k}}

            \mathcal{L}_{\textrm{KL}} &= \textrm{KLDiv}(\mathbf{P} \Vert
            \mathbf{S})
        """
        S_2 = S**2
        P = S_2 / S.sum(dim=1, keepdim=True)
        denom = P.sum(dim=2, keepdim=True)
        denom[S.sum(dim=2, keepdim=True) == 0.0] = 1.0
        P /= denom

        loss = KLDivLoss(reduction='batchmean', log_target=False)
        return loss(S.clamp(EPS).log(), P.clamp(EPS))

    def forward(self, x: Tensor, edge_index: Tensor, batch: Optional[Tensor] = None, gumbel_scale=1.) -> Tuple[Tensor, Tensor]:

        x, mask = to_dense_batch(x, batch)
        edge_index = unbatch_edge_index(edge_index, batch)

        (B, N, _), H, K = x.size(), self.heads, self.num_clusters2

        dist = torch.cdist(self.k.view(H * K, -1), x.view(B * N, -1), p=2)**2

        dist = (1. + dist / self.tau).pow(-(self.tau + 1.0) / 2.0)
        

        dist = dist.view(H, K, B, N).permute(2, 0, 3, 1)  # [B, H, N, K]

        S = dist / dist.sum(dim=-1, keepdim=True)

        
        S = S.sum(1)  # [B, N, K]
        
        S = S * mask.view(B, N, 1)
        for b in range(B):
            adj_t = torch.sparse_coo_tensor(edge_index[b], torch.ones(edge_index[b].shape[1]).to(S.device), size=(N,N)).to_dense()
            S[b,:,:] = torch.sparse.mm(adj_t, S[b,:,:])
            d = degree(edge_index[b].reshape(-1))/2
            d[d==0] = 1
            S[b,:d.shape[0],:] /= d.reshape(-1,1)
        S_raw = S
        
        if self.use_gumbel:
            gumbel = F.gumbel_softmax(S*gumbel_scale, dim=-1, tau=0.1)
            S = gumbel*(F.gumbel_softmax(S*gumbel_scale, dim=-1, hard=True, tau=0.1)).detach()
        else:
            gumbel = F.softmax(S, dim=-1)
            m = F.one_hot(gumbel.max(-1).indices, 3)
            S = gumbel*m
            
        x = self.lin(S.transpose(1, 2) @ x)

        return x[:,:self.num_clusters1,:], S, S_raw, dist, mask


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads}, '
                f'num_clusters={self.num_clusters_per_class})')


class IMem_Pool(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, dropout_att=0, num_clusters=2, heads=5, out_channels=16, explainable=False, gcn_normalize=False, lambda1=10, lambda2=0.01, **kwargs):
        super(IMem_Pool, self).__init__()
        self.hidden = hidden
        self.num_clusters = num_clusters
        self.num_classes = dataset.num_classes
        self.conv1 = GCNConv(dataset.num_features, hidden, normalize=gcn_normalize)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden, normalize=gcn_normalize))

        self.pool = IMPO(in_channels=num_layers*hidden, out_channels=out_channels, num_clusters_per_class=num_clusters, heads=heads, num_classes=dataset.num_classes, lambda1=lambda1, lambda2=lambda2, **kwargs)

        self.lin1 = Linear(num_clusters*dataset.num_classes*out_channels, hidden) # 2*hidden due to readout layer
        self.lin2 = Linear(hidden, dataset.num_classes)
        if explainable:
            self.expl_lin = Linear(num_clusters*heads*(dataset.num_classes+1), dataset.num_classes)
        self.reset_parameters()

        self.batch_norm = torch.nn.BatchNorm1d(num_layers*hidden)

        self.dropout_att = dropout_att



    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.pool.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def freeze_parameters(self, val=False):
        self.conv1.requires_grad_ = val 
        self.pool.requires_grad_ = val 
        for conv in self.convs:
            conv.requires_grad_ = val 
        lin1.requires_grad_ = val 
        lin2.requires_grad_ = val 




    def forward(self, data, gumbel_scale=1.):
        x, edge_index, edge_weight, batch, y = data.x, data.edge_index, data.edge_weight, data.batch, data.y

        x = F.relu(self.conv1(x, edge_index))

        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x=x, edge_index=edge_index, edge_weight=edge_weight))
            xs.append(x)

        x = torch.cat(xs, 1)


        x,S,S_raw,dist, mask = self.pool(x=self.batch_norm(x), edge_index=edge_index, batch=batch, gumbel_scale=gumbel_scale)
        total_loss = self.pool.impo_loss(S, S_raw, y, mask)
        
        x = x.reshape(x.shape[0],-1)
        

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_att, training=self.training)
        x = self.lin2(x)
        out = F.log_softmax(x, dim=-1)
        return out, total_loss


    def embs(self, data):
        x, edge_index, edge_weight, batch, y = data.x, data.edge_index, data.edge_weight, data.batch, data.y

        x = F.relu(self.conv1(x, edge_index))

        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x=x, edge_index=edge_index, edge_weight=edge_weight))
            xs.append(x)

        x = torch.cat(xs, 1)
        x=self.batch_norm(x)
        return x, edge_index, edge_weight, batch, y 

    def project(self, loader):
        H = self.pool.heads
        K = self.pool.num_clusters2
        best_dists = torch.ones(H, K)*1e8
        best_dists = best_dists.to(self.pool.k.device)
        new_k = torch.zeros_like(self.pool.k)
        new_k = new_k.to(self.pool.k.device)
        ks = torch.arange(K).to(self.pool.k.device)

        
        for batch in loader:
            x, edge_index, edge_weight, batch, y = self.embs(batch.to(self.pool.k.device))
            

            if x.dim() <= 2:
                x, mask = to_dense_batch(x, batch)
            elif mask is None:
                mask = x.new_ones((x.size(0), x.size(1)), dtype=torch.bool)

            (B, N, _) = x.size()

            dist = torch.cdist(self.pool.k.view(H * K, -1), x.view(B * N, -1), p=2)**2
            
            dist = (1. + dist / self.pool.tau).pow(-(self.pool.tau + 1.0) / 2.0)
            

            dist = dist.view(H, K, B, N)
            
            for yi in range(self.num_classes+1):
                try:
                    ks_cond = (self.num_clusters*yi<=ks) & (ks<self.num_clusters*(yi+1))
                    if yi == self.num_classes:
                        min_vals, min_nodes = dist[:,ks_cond,:,:].view(H,ks_cond.sum(),-1).min(-1)
                    else:
                        min_vals, min_nodes = dist[:,ks_cond,:,:][:,:,y==yi,:].view(H,ks_cond.sum(),-1).min(-1)
                    best_dists[:, ks_cond] = best_dists[:, ks_cond]*(1-(min_vals < best_dists[:, ks_cond]).float()) + min_vals*(min_vals < best_dists[:, ks_cond]).float()


                    if yi == self.num_classes:
                        new_k[:, ks_cond] = self.pool.k.data[:, ks_cond]*(1-(min_vals < best_dists[:, ks_cond]).float().unsqueeze(-1))+ x.view(B*N,-1)[min_nodes,:]*(min_vals < best_dists[:, ks_cond]).float().unsqueeze(-1)
                    else:
                        new_k[:, ks_cond] = self.pool.k.data[:, ks_cond]*(1-(min_vals < best_dists[:, ks_cond]).float().unsqueeze(-1))+ x[y==yi].view(-1,x.shape[-1])[min_nodes,:]*(min_vals < best_dists[:, ks_cond]).float().unsqueeze(-1)
                except: pass

                
        self.pool.k.data = new_k




    def forward_expl(self, data):
        x, edge_index, edge_weight, batch, y = data.x, data.edge_index, data.edge_weight, data.batch, data.y

        x = F.relu(self.conv1(x, edge_index))

        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x=x, edge_index=edge_index, edge_weight=edge_weight))
            xs.append(x)

        x = torch.cat(xs, 1)
        x,S,S_raw,dist, mask = self.pool(x=self.batch_norm(x), edge_index=edge_index, batch=batch, gumbel_scale=gumbel_scale)
        dist = dist / dist.sum(dim=-1, keepdim=True)
        dist = dist.permute(0,1,3,2)
        B, H, K, N = dist.shape
        dist = dist.reshape(B, H*K, N).permute(0,2,1)
        dist = dist  * mask.view(B, N, 1)
        x=dist.sum(1)
        x = self.expl_lin(x)
        out = F.log_softmax(x, dim=-1)
        return out

    def forward_activations(self, data, boolean=True, gumbel_scale=1.):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch

        x = F.relu(self.conv1(x, edge_index))

        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x=x, edge_index=edge_index, edge_weight=edge_weight))
            xs.append(x)

        x = torch.cat(xs, 1)

        x,S,S_raw,dist, mask = self.pool(x=self.batch_norm(x), edge_index=edge_index, batch=batch, gumbel_scale=gumbel_scale)
        
        out = (S.sum(1).cpu())/((S>0).int().sum(1).cpu()+1e-8)
        

        if boolean:
            out = ((S>0).cpu().int() * mask.unsqueeze(-1).float().cpu()).sum(1)
            
        return out, S, mask
