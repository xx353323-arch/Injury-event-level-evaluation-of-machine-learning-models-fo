import torch
import torch.nn as nn
from tca_gnn import TCAGNN, build_nvg_batch as _nvg_batch
from lmvg_tuned import LMVGTuned, hamming_distance


class GraphBackendWrapperTuned(nn.Module):
    def __init__(self, base_model: TCAGNN, backend="nvg", n_ch=10, T=14):
        super().__init__()
        self.base = base_model
        self.backend = backend
        if backend == "lmvg":
            self.lmvg = LMVGTuned(n_ch=n_ch, T=T)
        else:
            self.lmvg = None
        self._last_stats = {}
        self._last_hard_adj = None
        self._gumbel_temp = 1.0

    def set_gumbel_temp(self, t):
        self._gumbel_temp = float(t)

    def lmvg_param_names(self):
        if self.lmvg is None:
            return set()
        return {"lmvg.tau", "lmvg.view_logits"}

    def split_params(self):
        lmvg_set = self.lmvg_param_names()
        lmvg_params, base_params = [], []
        for n, p in self.named_parameters():
            if n in lmvg_set:
                lmvg_params.append(p)
            else:
                base_params.append(p)
        return base_params, lmvg_params

    def build_graph(self, x_btc):
        if self.backend == "nvg":
            adj = _nvg_batch(x_btc)
            hard = (adj > 0.5).float().detach()
            stats = dict(
                tau_mean=0.0, tau_std=0.0,
                view_w=[1.0, 0.0, 0.0],
                sparsity=float(hard.mean().item()),
                view_entropy=0.0, gumbel_temp=0.0,
                tau_shift=0.0, vl_shift=0.0,
            )
            self._last_stats = stats
            self._last_hard_adj = hard
            return adj, stats
        adj, stats = self.lmvg(x_btc, temperature=self._gumbel_temp)
        self._last_stats = stats
        self._last_hard_adj = self.lmvg._last_adj
        return adj, stats

    def forward(self, x_btc):
        B, T, C = x_btc.shape
        adj, _ = self.build_graph(x_btc)
        ch_embeds = []
        for c in range(C):
            h = self.base.proj(x_btc[:, :, c : c + 1])
            h = self.base.pos(h)
            h = self.base.intra[c](h, adj[:, c])
            ch_embeds.append(h.mean(dim=1, keepdim=True))
        h_bcd = torch.cat(ch_embeds, dim=1)
        h_bcd = self.base.inter(h_bcd)
        pooled = h_bcd.mean(dim=1)
        logit = self.base.head(pooled).squeeze(-1)
        return logit

    def extract_embedding(self, x_btc):
        B, T, C = x_btc.shape
        adj, _ = self.build_graph(x_btc)
        ch_embeds = []
        for c in range(C):
            h = self.base.proj(x_btc[:, :, c : c + 1])
            h = self.base.pos(h)
            h = self.base.intra[c](h, adj[:, c])
            ch_embeds.append(h.mean(dim=1, keepdim=True))
        h_bcd = torch.cat(ch_embeds, dim=1)
        h_bcd = self.base.inter(h_bcd)
        return h_bcd.mean(dim=1)


def make_model_tuned(backend="nvg", n_ch=10, T=14):
    base = TCAGNN(n_ch=n_ch, T=T)
    return GraphBackendWrapperTuned(base, backend=backend, n_ch=n_ch, T=T)
