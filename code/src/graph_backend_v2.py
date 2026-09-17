import torch
import torch.nn as nn
import torch.nn.functional as F
from tca_gnn import TCAGNN, build_nvg_batch as _nvg_batch
from lmvg_v2 import LMVGv2, hamming_distance


def _soft_graph_attn(h_btd, adj_btt, attn_module, norm_module):
    B, T, D = h_btd.shape
    causal_mask = torch.triu(torch.ones(T, T, device=h_btd.device), diagonal=1).bool()
    bias = torch.log(adj_btt.clamp_min(1e-6))
    bias = bias.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
    attn_bias = bias.unsqueeze(1).expand(B, attn_module.num_heads, T, T).reshape(B * attn_module.num_heads, T, T)
    out, _ = attn_module(h_btd, h_btd, h_btd, attn_mask=attn_bias, need_weights=False)
    return norm_module(h_btd + out)


class GraphBackendWrapperV2(nn.Module):
    def __init__(self, base_model: TCAGNN, backend="nvg", n_ch=10, T=14):
        super().__init__()
        self.base = base_model
        self.backend = backend
        if backend == "lmvg":
            self.lmvg = LMVGv2(n_ch=n_ch, T=T)
        else:
            self.lmvg = None
        self._last_stats = {}
        self._last_hard_adj = None
        self._gumbel_temp = 1.0

    def set_gumbel_temp(self, t):
        self._gumbel_temp = float(t)

    def set_hard(self, flag):
        if self.lmvg is not None:
            self.lmvg.set_hard(flag)

    def lmvg_param_names(self):
        if self.lmvg is None:
            return set()
        return {"lmvg.tau", "lmvg.view_logits"}

    def build_graph(self, x_btc):
        if self.backend == "nvg":
            adj = _nvg_batch(x_btc)
            hard = (adj > 0.5).float().detach()
            stats = dict(
                tau_mean=0.0, tau_std=0.0,
                view_w=[1.0, 0.0, 0.0],
                sparsity=float(hard.mean().item()),
                view_entropy=0.0, gumbel_temp=0.0,
                tau_shift=0.0, vl_shift=0.0, hard_mode=False,
            )
            self._last_stats = stats
            self._last_hard_adj = hard
            return adj, stats
        adj, stats = self.lmvg(x_btc, temperature=self._gumbel_temp)
        self._last_stats = stats
        self._last_hard_adj = self.lmvg._last_adj
        return adj, stats

    def _encode(self, x_btc, adj):
        B, T, C = x_btc.shape
        ch_embeds = []
        for c in range(C):
            h = self.base.proj(x_btc[:, :, c : c + 1])
            h = self.base.pos(h)
            h = self.base.intra[c](h, adj[:, c])
            ch_embeds.append(h.mean(dim=1, keepdim=True))
        h_bcd = torch.cat(ch_embeds, dim=1)
        h_bcd = self.base.inter(h_bcd)
        return h_bcd

    def forward(self, x_btc):
        adj, _ = self.build_graph(x_btc)
        h_bcd = self._encode(x_btc, adj)
        pooled = h_bcd.mean(dim=1)
        logit = self.base.head(pooled).squeeze(-1)
        return logit

    def extract_embedding(self, x_btc):
        adj, _ = self.build_graph(x_btc)
        h_bcd = self._encode(x_btc, adj)
        return h_bcd.mean(dim=1)


def make_model_v2(backend="nvg", n_ch=10, T=14):
    base = TCAGNN(n_ch=n_ch, T=T)
    return GraphBackendWrapperV2(base, backend=backend, n_ch=n_ch, T=T)
