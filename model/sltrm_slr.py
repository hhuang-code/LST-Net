import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.pointops.functions import pointops
# from lib.soft.kernel.subtraction import subtraction_gaussian_kernel

import pdb

EPSILON = 1e-6


def subtraction_gaussian_kernel_torch(q, k):
    # [B, H, H1*W1, C] @ [C, H2*W2] -> [B, H, H1*W1, H2*W2]
    matA_square = q ** 2. @ torch.ones(k.shape[-2:]).cuda()
    # [H1*W1, C] @ [B, H, C, H2*W2] -> [B, H, H1*W1, H2*W2]
    matB_square = torch.ones(q.shape[-2:]).cuda() @ k ** 2.

    out = matA_square + matB_square - 2. * (q @ k)

    return out


def newton_inv(mat, max_iter=20):
        P = mat
        I = torch.eye(mat.size(-1), device=mat.device)
        alpha = 2 / (torch.max(torch.sum(mat, dim=-1)) ** 2)
        beta = 0.5
        V = alpha * P
        pnorm = torch.max(torch.sum(torch.abs(I - torch.matmul(P, V)), dim=-2))
        err_cnt = 0
        while pnorm > 1.01 and err_cnt < 10:
            alpha *= beta
            V = alpha * P
            pnorm = torch.max(torch.sum(torch.abs(I - torch.matmul(P, V)), dim=-2))
            err_cnt += 1

        for i in range(max_iter):
            V = 2 * V - V @ P @ V

        return V


def batched_index_select(input, dim, index):
    """
    :param input: B x * x ... x *
    :param dim: 0 < scalar
    :param index: B x M
    :return:
    """
    views = [input.shape[0]] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)

    return torch.gather(input, dim, index)


class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16, landmark_ratio=0.125, lowrank=True,
                 sparse=True, sparse_ratio=0.75, vis_attn=False, block_id=0, kernel_method='cuda'):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample  # k-nearest neighbors

        assert round(landmark_ratio * self.nsample) == landmark_ratio * self.nsample
        self.landmark_ratio = landmark_ratio

        self.lowrank = lowrank

        self.sparse = sparse
        self.sparse_ratio = sparse_ratio

        self.vis_attn = vis_attn
        self.block_id = block_id

        self.n_landmarks = int(landmark_ratio * self.nsample)
        self.n_sparsities = int(math.sqrt(1 - self.sparse_ratio) * self.nsample)

        self.linear_c = nn.Linear(in_planes, mid_planes)    # for center
        self.linear_n = nn.Linear(in_planes, mid_planes)    # for neighbors
        self.linear_v = nn.Linear(in_planes, out_planes)

        self.Qnorm_act = nn.Sequential(nn.Linear(in_planes, out_planes, bias=False),
                                       nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))

        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True),
                                      nn.Linear(3, out_planes))

        self.linear_w = nn.Sequential(nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True),
                                      nn.Linear(mid_planes, mid_planes // share_planes),
                                      nn.BatchNorm1d(mid_planes // share_planes), nn.ReLU(inplace=True),
                                      # nn.Linear(out_planes // share_planes, out_planes // share_planes))
                                      nn.Linear(out_planes // share_planes, out_planes))

        # self.softmax = nn.Softmax(dim=1)

        # if kernel_method == 'torch':
        #     self.kernel_function = subtraction_gaussian_kernel_torch
        # elif kernel_method == 'cuda':
        #     self.kernel_function = subtraction_gaussian_kernel
        # else:
        #     assert False, 'Please choose kernel method from torch and cuda.'

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)

        # x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)  # (n, c)
        # x_q, x_v = self.linear_q(x), self.linear_v(x)  # (n, c)
        x_c, x_n, x_v = self.linear_c(x), self.linear_n(x), self.linear_v(x)  # (n, c)

        # x_n, idx = pointops.queryandgroup(self.nsample, p, p, x_n, None, o, o, use_xyz=True, return_index=True)  # (n, nsample, 3+c)
        # x_n, idx = pointops.queryandgroup(self.nsample, p, p, x_n, None, o, o, use_xyz=True, return_index=True)  # (n, nsample, 3+c)
        x_n = pointops.queryandgroup(self.nsample, p, p, x_n, None, o, o, use_xyz=True, return_index=False)  # (n, nsample, 3+c)
        x_v = pointops.queryandgroup(self.nsample, p, p, x_v, None, o, o, use_xyz=False, return_index=False)  # (n, nsample, c)
        # p_n, x_n = x_n[:, :, 0:3], x_n[:, :, 3:]  # split xyz and feat
        p_n, x_n = x_n[:, :, 0:3], x_n[:, :, 3:]  # split xyz and feat

        # positional encoding, the subtraction operation is done in queryandgroup
        for i, layer in enumerate(self.linear_p):
            # (n, nsample, c)
            p_n = layer(p_n.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(p_n)

        # x_q_rel = x_n - x_q.unsqueeze(1) + p_n  # (n, nsample, c)
        x_q = x_c.unsqueeze(1) - x_n + p_n  # (n, nsample, c
        x_v = x_v + p_n     # (n, nsample, c)

        # group neighboring points to form landmarks
        x_landmarks = torch.stack(torch.split(x_q, self.nsample // self.n_landmarks, dim=1), dim=1)  # (n, n_landmarks, m, c)
        x_landmarks = torch.max(x_landmarks, dim=2)[0]  # (n, n_landmarks, c)

        for i, layer in enumerate(self.Qnorm_act):
            x_landmarks = layer(x_landmarks.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 2 == 1 else layer(x_landmarks)

        # reshape Q_landmarks to (b=n, n_heads=1, n_landmarks=m, c) to be compatible with kernel_function
        # Q, Q_landmarks = x_q_rel.unsqueeze(1), x_landmarks.unsqueeze(1)
        Q, Q_landmarks = x_q.unsqueeze(1), x_landmarks.unsqueeze(1)

        # Q = Q / (torch.norm(Q, p=2, dim=-1, keepdim=True) + EPSILON)
        Q = F.normalize(Q, p=2, dim=-1)

        # Q_landmarks = Q_landmarks / (torch.norm(Q_landmarks, p=2, dim=-1, keepdim=True) + EPSILON)
        Q_landmarks = F.normalize(Q_landmarks, p=2, dim=-1)

        K_landmarks = Q_landmarks

        kernel_1_ = torch.matmul(Q, K_landmarks.transpose(-1, -2).contiguous())

        kernel_2_ = torch.matmul(Q_landmarks, K_landmarks.transpose(-1, -2).contiguous())

        # kernel_1_ = 1 - self.kernel_function(Q, K_landmarks.transpose(-1, -2).contiguous()) / 2
        #
        # kernel_2_ = 1 - self.kernel_function(Q_landmarks, K_landmarks.transpose(-1, -2).contiguous()) / 2

        # kernel_1_ = self.kernel_function(Q, K_landmarks.transpose(-1, -2).contiguous())  # (b=n, n_heads=1, nsample, n_landmarks)
        # kernel_1_ = torch.exp(-kernel_1_ / (2 * math.sqrt(math.sqrt(x_q.shape[-1]))))
        #
        # kernel_2_ = self.kernel_function(Q_landmarks, K_landmarks.transpose(-1, -2).contiguous())
        # kernel_2_ = torch.exp(-kernel_2_ / (2 * math.sqrt(math.sqrt(x_q.shape[-1]))))

        kernel_1_ = (kernel_1_ + 1) / 2
        kernel_2_ = (kernel_2_ + 1) / 2

        kernel_3_ = kernel_1_.transpose(-1, -2)  # due to Q_landmarks = K_landmarks

        if self.lowrank:
            w = torch.matmul(torch.matmul(kernel_1_, newton_inv(kernel_2_)), torch.matmul(kernel_3_, x_v.unsqueeze(1))).squeeze(1)  # (b=n, nsample, c)
        else:
            w = torch.zeros((kernel_1_.shape[0], self.nsample, x_v.shape[2]), device=x_v.device)

        # add sparse attention
        if self.sparse:
            c_id = torch.arange(self.n_landmarks, device=kernel_1_.device).repeat_interleave(self.nsample // self.n_landmarks)
            c_id = c_id + self.n_landmarks * torch.arange(self.nsample, device=kernel_1_.device)
            c_id = c_id.unsqueeze(0).expand(kernel_1_.shape[0], self.nsample)
            similarity = batched_index_select(kernel_1_.squeeze(1).view(kernel_1_.shape[0], -1), dim=1, index=c_id)     # (n, nsample)
            topk_idx = torch.topk(similarity, dim=-1, k=self.n_sparsities, largest=False)[1]  # (n, topk)

            # similarity = torch.einsum('nkc,nc->nk', Q[:, 0, ...], Q[:, 0, 0, :])    # (n, nsample)

            # x_n_group = x_n_group / (torch.norm(x_n_group, p=2, dim=-1, keepdim=True) + EPSILON)
            # similarity = torch.einsum('nkc,nc->nk', x_n_group, x_n_group[:, 0, :])  # (n, nsample)

            # x_n = x_n / (torch.norm(x_n, p=2, dim=-1, keepdim=True) + EPSILON)
            # x_q = x_q / (torch.norm(x_q, p=2, dim=-1, keepdim=True) + EPSILON)
            # similarity = torch.einsum('nkc,nc->nk', x_n, x_q)  # (n, nsample)
            # topk_idx = torch.topk(similarity, dim=-1, k=16, largest=True)[1]  # (n, topk)
            # topk_idx = torch.topk(similarity, dim=-1, k=self.n_sparsities, largest=True)[1]  # (n, topk)

            # attn = torch.matmul(torch.matmul(kernel_1_, newton_inv(kernel_2_)), kernel_3_)  # (n, 1, nsample, nsample)
            # mask = torch.ones_like(similarity).scatter_(dim=1, index=topk_idx, value=0).bool()  # (n, nsample)
            # mask = torch.bitwise_or(mask.unsqueeze(2), mask.unsqueeze(1))
            # w = torch.matmul(attn.squeeze(1) * mask, x_v)  # (n, nsample, c)

            Q_ns = batched_index_select(Q.squeeze(1), dim=1, index=topk_idx)  # (n, topk, c)
            kernel_s_ = torch.matmul(Q_ns, Q_ns.transpose(1, 2).contiguous())  # (n, topk, topk)
            kernel_s_ = (kernel_s_ + 1) / 2
            x_v_s = batched_index_select(x_v, dim=1, index=topk_idx)  # (n, topk, c)
            w_s = torch.matmul(kernel_s_, x_v_s)  # (n, topk, c)

            w = torch.scatter_add(w, dim=1, index=topk_idx.unsqueeze(-1).expand_as(w_s), src=w_s)

        # visualize attention weights
        if self.vis_attn:
            self.attn = []
            # random pick one points in each point cloud
            attn = torch.matmul(torch.matmul(kernel_1_, newton_inv(kernel_2_)), kernel_3_)
            for i in range(o.shape[0]):
                if i == 0:
                    pt_idx = torch.randint(low=0, high=o[i], size=(1,))
                else:
                    pt_idx = torch.randint(low=o[i - 1], high=o[i], size=(1,))
                # image shape should be CHW
                self.attn.append({'block_id': self.block_id, 'pt_idx': pt_idx, 'attn': attn[pt_idx].squeeze(0).detach().cpu().numpy()})

        for i, layer in enumerate(self.linear_w):
            w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)

        x = w.max(dim=1)[0]     # (n, c)

        return x


# class PointTransformerLayer(nn.Module):
#     def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
#         super().__init__()
#         self.mid_planes = mid_planes = out_planes // 1
#         self.out_planes = out_planes
#         self.share_planes = share_planes
#         self.nsample = nsample  # k-nearest neighbors
#         self.linear_q = nn.Linear(in_planes, mid_planes)
#         self.linear_k = nn.Linear(in_planes, mid_planes)
#         self.linear_v = nn.Linear(in_planes, out_planes)
#         self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True),
#                                       nn.Linear(3, out_planes))
#         self.linear_w = nn.Sequential(nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True),
#                                       nn.Linear(mid_planes, mid_planes // share_planes),
#                                       nn.BatchNorm1d(mid_planes // share_planes), nn.ReLU(inplace=True),
#                                       nn.Linear(out_planes // share_planes, out_planes // share_planes))
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, pxo) -> torch.Tensor:
#         p, x, o = pxo  # (n, 3), (n, c), (b)
#         x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)  # (n, c)
#         x_k = pointops.queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=True)  # (n, nsample, 3+c)
#         x_v = pointops.queryandgroup(self.nsample, p, p, x_v, None, o, o, use_xyz=False)  # (n, nsample, c)
#         p_k, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]  # split xyz and feat
#
#         # Eq.(4), the subtraction operation is done in queryandgroup
#         for i, layer in enumerate(self.linear_p):
#             # (n, nsample, c)
#             p_k = layer(p_k.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(p_k)
#
#         # ϕ(xi)−ψ(xj)+δ in Eq.(3)
#         w = x_k - x_q.unsqueeze(1) + p_k.view(p_k.shape[0], p_k.shape[1], self.out_planes // self.mid_planes,
#                                               self.mid_planes).sum(2)  # (n, nsample, c)
#
#         # γ(ϕ(xi)−ψ(xj)+δ) in Eq.(3)
#         for i, layer in enumerate(self.linear_w):
#             w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)
#
#         # ρ(γ(ϕ(xi)−ψ(xj)+δ)) in Eq.(3)
#         w = self.softmax(w)  # (n, nsample, c)
#
#         n, nsample, c = x_v.shape
#         s = self.share_planes
#         # Eq. (3)
#         x = ((x_v + p_k).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)
#
#         return x  # (n, c)


class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample

        if stride != 1:
            self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)   # pool over k-nearest neighbors
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)

        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)

        if self.stride != 1:
            # compute new offset
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i - 1].item()) // self.stride
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o)

            idx = pointops.furthestsampling(p, o, n_o)  # (m,)
            n_p = p[idx.long(), :]  # (m, 3)

            x = pointops.queryandgroup(self.nsample, p, n_p, x, None, o, n_o, use_xyz=True)  # (m, nsample, 3+c)
            x = self.relu(self.bn(self.linear(x).transpose(1, 2).contiguous()))  # (m, c, nsample)
            y = self.pool(x).squeeze(-1)  # (m, c)

            p, o = n_p, n_o
        else:
            y = self.relu(self.bn(self.linear(x)))  # (n, c)

        return [p, y, o]


# class TransitionUp(nn.Module):
#     def __init__(self, in_planes, out_planes=None):
#         super().__init__()
#         if out_planes is None:
#             self.linear1 = nn.Sequential(nn.Linear(2 * in_planes, in_planes),
#                                          nn.BatchNorm1d(in_planes), nn.ReLU(inplace=True))
#             self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True))
#         else:
#             self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes),
#                                          nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
#             self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes),
#                                          nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
#
#     def forward(self, pxo1, pxo2=None):
#         if pxo2 is None:
#             _, x, o = pxo1  # (n, 3), (n, c), (b)
#             x_tmp = []
#             for i in range(o.shape[0]):
#                 if i == 0:
#                     s_i, e_i, cnt = 0, o[0], o[0]
#                 else:
#                     s_i, e_i, cnt = o[i - 1], o[i], o[i] - o[i - 1]
#                 x_b = x[s_i:e_i, :]
#                 x_b = torch.cat((x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1)
#                 x_tmp.append(x_b)
#             x = torch.cat(x_tmp, 0)
#             x = self.linear1(x)
#         else:
#             # number of points in p1 is larger than that in p2
#             p1, x1, o1 = pxo1
#             p2, x2, o2 = pxo2
#             x = self.linear1(x1) + pointops.interpolation(p2, p1, self.linear2(x2), o2, o1)  # k-nearest neighbors in p2 for p1
#
#         return x


class PointTransformerBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16, landmark_ratio=0.125, lowrank=True, sparse=True, sparse_ratio=0.75, vis_attn=False, block_id=0):
        super(PointTransformerBlock, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)

        # self.transformer = PointTransformerLayer(planes, planes, share_planes, nsample)
        kwargs = {'landmark_ratio': landmark_ratio, 'lowrank': lowrank, 'sparse': sparse,
                  'sparse_ratio': sparse_ratio, 'vis_attn': vis_attn, 'block_id': block_id}
        self.transformer = PointTransformerLayer(planes, planes, share_planes, nsample, **kwargs)

        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)

        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        identity = x

        x = self.relu(self.bn1(self.linear1(x)))

        x = self.relu(self.bn2(self.transformer([p, x, o])))

        x = self.bn3(self.linear3(x))
        x += identity
        x = self.relu(x)

        return [p, x, o]


class SparseLowrankTransformerSeg(nn.Module):
    def __init__(self, block, blocks, c=6, k=13, lowrank=True, sparse=True, vis_attn=False):
        super().__init__()
        self.c = c

        # self.in_planes, planes = c, [32, 64, 128, 256, 512]
        self.in_planes, planes = c, [16, 32, 64, 128, 256]
        # fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 1

        # stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]
        # stride, nsample = [1, 4, 4, 4, 4], [16, 16, 16, 16, 16]
        # stride, nsample = [1, 4, 4, 4, 4], [32, 32, 32, 32, 32]
        stride, nsample = [1, 4, 4, 4, 4], [48, 48, 48, 48, 48]
        # stride, nsample = [1, 4, 4, 4, 4], [64, 64, 64, 64, 64]
        # stride, nsample = [1, 4, 4, 4, 4], [80, 80, 80, 80, 80]

        landmark_ratio = 1 / 16.0    # n_landmarks = int(nsample * landmark_ratio)
        sparse_ratio = 0.93         # number of points for non-zero entries = int(nsample * sqrt(1 - sparse_ratio))

        for block_id in range(1, 6):
            kwargs = {'landmark_ratio': landmark_ratio, 'lowrank': lowrank, 'sparse': sparse,
                      'sparse_ratio': sparse_ratio, 'vis_attn': vis_attn, 'block_id': block_id}
            setattr(self, 'enc' + str(block_id), self._make_enc(
                block, planes[block_id - 1], blocks[block_id - 1], share_planes, stride[block_id - 1], nsample[block_id - 1], **kwargs))

        # self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes, stride=stride[0], nsample=nsample[0])  # N / 1
        # self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes, stride=stride[1], nsample=nsample[1])  # N / 4
        # self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes, stride=stride[2], nsample=nsample[2])  # N / 16
        # self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes, stride=stride[3], nsample=nsample[3])  # N / 64
        # self.enc5 = self._make_enc(block, planes[4], blocks[4], share_planes, stride=stride[4], nsample=nsample[4])  # N / 256

        # self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes, stride[0], nsample[0], landmark_ratio, self.sparse, self.vis_attn)
        # self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes, stride[1], nsample[1], landmark_ratio, self.sparse, self.vis_attn)
        # self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes, stride[2], nsample[2], landmark_ratio, self.sparse, self.vis_attn)
        # self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes, stride[3], nsample[3], landmark_ratio, self.sparse, self.vis_attn)
        # self.enc5 = self._make_enc(block, planes[4], blocks[4], share_planes, stride[4], nsample[4], landmark_ratio, self.sparse, self.vis_attn)

        # self.dec5 = self._make_dec(block, planes[4], 1, share_planes, nsample=nsample[4], is_head=True)  # transform p5
        # self.dec4 = self._make_dec(block, planes[3], 1, share_planes, nsample=nsample[3])  # fusion p5 and p4
        # self.dec3 = self._make_dec(block, planes[2], 1, share_planes, nsample=nsample[2])  # fusion p4 and p3
        # self.dec2 = self._make_dec(block, planes[1], 1, share_planes, nsample=nsample[1])  # fusion p3 and p2
        # self.dec1 = self._make_dec(block, planes[0], 1, share_planes, nsample=nsample[0])  # fusion p2 and p1

        # self.cls = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], k))

        self.cls = nn.Sequential(nn.Linear(sum(planes[1:]), planes[3]), nn.BatchNorm1d(planes[3]), nn.ReLU(inplace=True),
                                 nn.Linear(planes[3], planes[2]), nn.BatchNorm1d(planes[2]), nn.ReLU(inplace=True),
                                 nn.Linear(planes[2], planes[1]), nn.BatchNorm1d(planes[1]), nn.ReLU(inplace=True),
                                 nn.Linear(planes[1], k))

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16, landmark_ratio=0.125,
                  lowrank=True, sparse=True, sparse_ratio=0.75, vis_attn=False, block_id=0):
        layers = []

        layers.append(TransitionDown(self.in_planes, planes * block.expansion, stride, nsample))

        self.in_planes = planes * block.expansion

        # add transformer blocks
        for _ in range(blocks):
            kwargs = {'landmark_ratio': landmark_ratio, 'lowrank': lowrank, 'sparse': sparse,
                      'sparse_ratio': sparse_ratio, 'vis_attn': vis_attn, 'block_id': block_id}
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample, **kwargs))

        return nn.Sequential(*layers)

    # def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
    #     layers = []
    #     layers.append(TransitionUp(self.in_planes, None if is_head else planes * block.expansion))
    #     self.in_planes = planes * block.expansion
    #
    #     # add transformer blocks
    #     for _ in range(blocks):
    #         layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
    #
    #     return nn.Sequential(*layers)

    def forward(self, pxmo):
        p0, x0, m0, o0 = pxmo  # (n, 3), (n, c), (n,), (b,)
        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)

        p1, x1, o1 = self.enc1([p0, x0, o0])    # N / 1
        p2, x2, o2 = self.enc2([p1, x1, o1])    # N / 4
        p3, x3, o3 = self.enc3([p2, x2, o2])    # N / 16
        p4, x4, o4 = self.enc4([p3, x3, o3])    # N / 64
        p5, x5, o5 = self.enc5([p4, x4, o4])    # N / 256

        p0_q = p0[m0.nonzero(as_tuple=True)]
        chunks = (o0.cpu().numpy().astype(int) - (np.concatenate([[0], o0.cpu().numpy().astype(int)[:-1]]))).tolist()
        o0_q = torch.cumsum(torch.cat([torch.sum(x, dim=0, keepdim=True) for x in torch.split(m0, chunks)]), dim=0).int()

        x2_q = pointops.interpolation(p2, p0_q, x2, o2, o0_q, k=3)  # k-nearest neighbors in p2 for p1, (m, c)
        x3_q = pointops.interpolation(p3, p0_q, x3, o3, o0_q, k=3)
        x4_q = pointops.interpolation(p4, p0_q, x4, o4, o0_q, k=3)
        x5_q = pointops.interpolation(p5, p0_q, x5, o5, o0_q, k=3)

        x_q = torch.cat([x2_q, x3_q, x4_q, x5_q], dim=-1)

        # x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        # x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1]
        # x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        # x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        # x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]

        # x = self.cls(x1)

        # avoid errors in batch norm for a single sample
        if x_q.shape[0] == 1:
            x = self.cls(x_q.repeat(2, 1)).split(x_q.shape[0], dim=0)[0]
        else:
            x = self.cls(x_q)

        return x


def sparse_lowrank_trm_seg(**kwargs):
    # model = SparseLowrankTransformerSeg(PointTransformerBlock, [1, 2, 3, 5, 2], **kwargs)
    model = SparseLowrankTransformerSeg(PointTransformerBlock, [1, 1, 1, 1, 1], **kwargs)
    return model


if __name__ == '__main__':

    from model.sltrm_slr import sparse_lowrank_trm_seg as Model

    feat_dim = 6
    classes = 13
    sparse = True
    vis_attn = False
    model = Model(c=feat_dim, k=classes, sparse=sparse, vis_attn=vis_attn).cuda()

    n = 40960 + 40960
    c = 3
    coord = torch.randn(n, 3).cuda()
    feat = torch.randn(n, 3).cuda()
    mask = torch.randint(0, 2, (n,)).cuda().bool()
    offset = torch.Tensor([40960, 40960 + 40960]).int().cuda()

    y = model([coord, feat, mask, offset])

    pdb.set_trace()


    print('Done.')


