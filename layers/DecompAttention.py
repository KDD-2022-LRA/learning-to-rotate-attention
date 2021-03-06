import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import numpy as np
import math
from math import sqrt, pi, log2
from utils.masking import TriangularCausalMask, ProbMask
import os

class TrendRefinement(nn.Module):
    def __init__(self, seq_len, d_model, n_heads, kernel_size=7, d_values=None):
        super(TrendRefinement, self).__init__()
        d_values = d_values or (d_model // n_heads)
        self.seq_len = seq_len
        D =  torch.matmul(self._gen_D_matrix(seq_len-1), self._gen_D_matrix(seq_len)) # the second-order difference matrix
        U, S, _ = torch.linalg.svd(torch.mm(D.T, D))
        self.register_buffer('U', U)
        self.register_buffer('S', S)
        self.lambda_projection = nn.Conv1d(d_values, 1, kernel_size, padding=kernel_size//2, padding_mode='replicate') # shared by multiple heads

    def forward(self, values):
        B, L, H, E = values.shape
        lambdas = 1. + F.elu(self.lambda_projection(values.permute(0,2,3,1).reshape(-1,E,L)).reshape(B,H,1,L)).permute(0,3,1,2).reshape(B,L,H) # B, L, H, 1
        s = torch.einsum('blh,l->bhl', lambdas, self.S)
        g = torch.diag_embed(1. / (1. + s))
        g = torch.einsum('im,bhmn,nj->bhij',self.U, g, self.U.transpose(-1, -2))
        values = torch.einsum('bhij,bjhd->bihd', g, values)
        return values.contiguous()

    def _gen_D_matrix(self, L):
        """calculate the first-order difference matrix for sequence of length L
        shape: (L-1, L)
        """
        D = torch.zeros(L - 1, L)
        D[:, 1:] = torch.eye(L - 1)
        D[:, :-1] -= torch.eye(L - 1)
        return D

class QuaternionAttention(nn.Module):
    def __init__(self, query_size, key_size, mask_flag=False, scale=None, attention_dropout=0.1, output_attention=False):
        super(QuaternionAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.query_size = query_size
        self.key_size = key_size

        query_pos = torch.arange(0.0, query_size, 1.0).view(-1, 1, 1)
        key_pos = torch.arange(0.0, key_size, 1.0).view(-1, 1, 1)
        self.register_buffer('query_pos', query_pos)
        self.register_buffer('key_pos', key_pos)

    def forward(self, queries, keys, values, query_omegas, query_thetas, key_omegas, key_thetas, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, _ = keys.shape
        _, _, _, M = query_omegas.shape

        # a quaternion version
        Q_angles = query_omegas * self.query_pos + query_thetas # (B, L, H, M)
        K_angles = key_omegas * self.key_pos + key_thetas # (B, S, H, M)
        
        Q_cos, Q_sin = Q_angles.cos(), Q_angles.sin() # (B, L, H, M)
        K_cos, K_sin = K_angles.cos(), K_angles.sin() # (B, S, H, M)

        Q_quaternion = torch.chunk(queries, 4, dim=-1) # (B, L, H, E//4) of 4
        K_quaternion = torch.chunk(keys, 4, dim=-1) # (B, S, H, E//4) of 4
        
        Q_rotation = torch.cat(
            [
                torch.einsum('blhe,blhm->blhme', Q_quaternion[0], Q_cos) - torch.einsum('blhe,blhm->blhme', Q_quaternion[1], Q_sin),
                torch.einsum('blhe,blhm->blhme', Q_quaternion[1], Q_cos) + torch.einsum('blhe,blhm->blhme', Q_quaternion[0], Q_sin),
                torch.einsum('blhe,blhm->blhme', Q_quaternion[2], Q_cos) + torch.einsum('blhe,blhm->blhme', Q_quaternion[3], Q_sin),
                torch.einsum('blhe,blhm->blhme', Q_quaternion[3], Q_cos) - torch.einsum('blhe,blhm->blhme', Q_quaternion[2], Q_sin),
            ], dim=-1
        ) # (B, L, H, M, E)

        K_rotation = torch.cat(
            [
                torch.einsum('bshe,bshm->bshme', K_quaternion[0], K_cos) - torch.einsum('bshe,bshm->bshme', K_quaternion[2], K_sin),
                torch.einsum('bshe,bshm->bshme', K_quaternion[1], K_cos) - torch.einsum('bshe,bshm->bshme', K_quaternion[3], K_sin),
                torch.einsum('bshe,bshm->bshme', K_quaternion[2], K_cos) + torch.einsum('bshe,bshm->bshme', K_quaternion[0], K_sin),
                torch.einsum('bshe,bshm->bshme', K_quaternion[3], K_cos) + torch.einsum('bshe,bshm->bshme', K_quaternion[1], K_sin),
            ], dim=-1
        ) # (B, S, H, M, E)
        
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhme,bshme->bhls", Q_rotation, K_rotation) / M

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class FullAttention(nn.Module):
    def __init__(self, query_size, key_size, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, query_omegas, query_thetas, key_omegas, key_thetas, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class LearningToRotateAttentionLayer(nn.Module):
    def __init__(self, attention, query_size, key_size, d_model, n_heads, moving_avg=25, period_type='variant', n_periods=2, d_keys=None,
                 d_values=None, is_trend_refinement=True):

        super(LearningToRotateAttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.n_heads = n_heads
        self.period_type = period_type
        self.n_periods = n_periods

        # series decompostion
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)

        # trend refinement
        self.is_trend_refinement = is_trend_refinement
        if self.is_trend_refinement:
            self.trend_refine = TrendRefinement(key_size, d_model, n_heads)

        # learning-to-rotate attention
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)

        kernel_size = 1
        padding = kernel_size // 2
        if period_type == 'variant':
            self.query_omega_projection = nn.Conv1d(d_model, n_periods * n_heads, kernel_size=kernel_size, padding=padding, padding_mode='zeros')
            self.key_omega_projection = nn.Conv1d(d_model, n_periods * n_heads, kernel_size=kernel_size, padding=padding, padding_mode='zeros')
        else:
            self.query_omega_projection = nn.Linear(d_model, n_periods * n_heads)
            self.key_omega_projection = nn.Linear(d_model, n_periods * n_heads)

        self.query_theta_projection = nn.Conv1d(d_model, n_periods * n_heads, kernel_size=kernel_size, padding=padding, padding_mode='zeros')
        self.key_theta_projection = nn.Conv1d(d_model, n_periods * n_heads, kernel_size=kernel_size, padding=padding, padding_mode='zeros')

        # regularization
        query_D_matrix = self._gen_D_matrix(query_size)
        key_D_matrix = self._gen_D_matrix(key_size)
        self.register_buffer('query_D_matrix', query_D_matrix)
        self.register_buffer('key_D_matrix', key_D_matrix)

        # output projection
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        
        
    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        if self.period_type == 'variant':
            query_omegas = F.relu(self.query_omega_projection(queries.transpose(1, 2))).transpose(1, 2).view(B, L, H, -1)
            key_omegas = F.relu(self.key_omega_projection(keys.transpose(1,2))).transpose(1, 2).view(B, S, H, -1)
        else:
            query_omegas = F.relu(self.query_omega_projection(torch.mean(queries, dim=1))).view(B, 1, H, -1).repeat(1, L, 1, 1)
            key_omegas = F.relu(self.key_omega_projection(torch.mean(keys, dim=1))).view(B, 1, H, -1).repeat(1, S, 1, 1)
    
        query_thetas = (F.tanh(self.query_theta_projection(queries.transpose(1, 2)).transpose(1, 2)) * pi).view(B, L, H, -1)
        key_thetas = (F.tanh(self.key_theta_projection(keys.transpose(1, 2)).transpose(1, 2)) * pi).view(B, S, H, -1)

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        queries, _ = self.decomp1(queries)
        keys, _ = self.decomp2(keys)
        values, values_trend = self.decomp3(values)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            query_omegas,
            query_thetas,
            key_omegas,
            key_thetas,
            attn_mask
        )

        if self.is_trend_refinement:
            out_trend = self.trend_refine(values_trend)
            out += out_trend

        out = out.view(B, L, -1)

        # calculate penalty of a single attention layer
        query_omegas_diff = torch.einsum('ji,bihm->bjhm', self.query_D_matrix, query_omegas)
        key_omegas_diff = torch.einsum('ji,bihm->bjhm', self.key_D_matrix, key_omegas)

        query_omegas_penalty = torch.sum(query_omegas_diff ** 2)
        key_omegas_penalty = torch.sum(key_omegas_diff ** 2)
        query_thetas_penalty = torch.sum(query_thetas ** 2)
        key_thetas_penalty = torch.sum(key_thetas ** 2)

        omegas_penalty = (query_omegas_penalty + key_omegas_penalty)
        thetas_penalty = (query_thetas_penalty + key_thetas_penalty)

        return self.out_projection(out), attn, omegas_penalty, thetas_penalty

    def _gen_D_matrix(self, L):
        """calculate the first-order difference matrix for sequence of length L
        """
        D = torch.zeros(L - 1, L)
        D[:, 1:] = torch.eye(L - 1)
        D[:, :-1] -= torch.eye(L - 1)
        return D


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        B, L, H, E = x.shape
        front = x[:, 0:1, :, :].repeat(1, (self.kernel_size - 1) // 2, 1, 1)
        end = x[:, -1:, :, :].repeat(1, (self.kernel_size - 1) // 2, 1, 1)
        x = torch.cat([front, x, end], dim=1).view(B, -1, H*E)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x.view(B, -1, H, E)


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

