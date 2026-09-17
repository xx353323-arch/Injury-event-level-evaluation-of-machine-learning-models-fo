import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _causal_visibility(x_btc):
    B, T, C = x_btc.shape
    device = x_btc.device
    y = x_btc.transpose(1, 2)
    t_i = torch.arange(T, device=device).view(T, 1, 1).float()
    t_j = torch.arange(T, device=device).view(1, T, 1).float()
    t_k = torch.arange(T, device=device).view(1, 1, T).float()
    y_i = y.unsqueeze(-1).unsqueeze(-1)
    y_j = y.unsqueeze(-2).unsqueeze(-1)
    y_k = y.unsqueeze(-2).unsqueeze(-2)
    diff = (t_j - t_i)
    slope = (y_j - y_i) / torch.clamp(diff, min=1e-6)
    line_k = y_i + slope * (t_k - t_i)
    between = ((t_k > t_i) & (t_k < t_j)).float()
    block = ((y_k >= line_k).float() * between).sum(dim=-1)
    any_block = (block > 0).float()
    causal = (t_j.squeeze(-1) > t_i.squeeze(-1)).float().unsqueeze(0).unsqueeze(0)
    visible = (1 - any_block) * causal
    return visible


def _corr_logits(x_btc):
    B, T, C = x_btc.shape
    y = x_btc.transpose(1, 2)
    yi = y.unsqueeze(-1)
    yj = y.unsqueeze(-2)
    sim = -(yi - yj).pow(2)
    std = sim.std(dim=(-1, -2), keepdim=True) + 1e-6
    return sim / std


def _temporal_logits(B, C, T, device, decay=3.0):
    t_i = torch.arange(T, device=device).view(T, 1).float()
    t_j = torch.arange(T, device=device).view(1, T).float()
    dec = -torch.abs(t_i - t_j) / decay
    return dec.view(1, 1, T, T).expand(B, C, T, T)


def cosine_temperature(ep, total, t_start=1.0, t_end=0.1):
    if total <= 1:
        return t_end
    r = min(max((ep - 1) / (total - 1), 0.0), 1.0)
    return t_end + 0.5 * (t_start - t_end) * (1.0 + math.cos(math.pi * r))


class LMVGTuned(nn.Module):
    def __init__(self, n_ch=10, T=14, tau_init=0.0, hard=True):
        super().__init__()
        self.n_ch = n_ch
        self.T = T
        self.tau = nn.Parameter(torch.full((n_ch,), float(tau_init)))
        self.view_logits = nn.Parameter(torch.zeros(3))
        self.hard = hard
        self._last_adj = None
        self.register_buffer("_tau_init", self.tau.detach().clone())
        self.register_buffer("_vl_init", self.view_logits.detach().clone())

    def _gumbel_edge(self, logits, tau_ch, temperature):
        z = logits - tau_ch
        stacked = torch.stack([z, torch.zeros_like(z)], dim=-1)
        m = F.gumbel_softmax(stacked, tau=temperature, hard=self.hard, dim=-1)
        return m[..., 0]

    def forward(self, x_btc, temperature=1.0):
        B, T, C = x_btc.shape
        device = x_btc.device
        v_causal = _causal_visibility(x_btc)
        v_corr = _corr_logits(x_btc)
        v_temp = _temporal_logits(B, C, T, device)
        tau_ch = self.tau.view(1, C, 1, 1)
        m_corr = self._gumbel_edge(v_corr, tau_ch, temperature)
        m_temp = self._gumbel_edge(v_temp, tau_ch, temperature)
        w = F.softmax(self.view_logits, dim=0)
        fused = w[0] * v_causal + w[1] * m_corr + w[2] * m_temp
        eye = torch.eye(T, device=device).view(1, 1, T, T)
        fused = fused * (1 - eye) + eye
        hard_adj = (fused > 0.5).float().detach()
        self._last_adj = hard_adj
        with torch.no_grad():
            ent = -(w * torch.log(w.clamp_min(1e-12))).sum().item()
            tau_shift = (self.tau - self._tau_init).norm().item()
            vl_shift = (self.view_logits - self._vl_init).norm().item()
        stats = dict(
            tau_mean=float(self.tau.detach().mean().item()),
            tau_std=float(self.tau.detach().std().item()),
            view_w=w.detach().cpu().numpy().tolist(),
            sparsity=float(hard_adj.mean().item()),
            view_entropy=float(ent),
            gumbel_temp=float(temperature),
            tau_shift=float(tau_shift),
            vl_shift=float(vl_shift),
        )
        return fused, stats


def hamming_distance(a_prev, a_curr):
    if a_prev is None or a_curr is None:
        return 0.0
    if a_prev.shape != a_curr.shape:
        return 1.0
    return float((a_prev != a_curr).float().mean().item())
