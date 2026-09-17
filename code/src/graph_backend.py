import torch
import torch.nn as nn
from tca_gnn import TCAGNN, build_nvg_batch as _nvg_batch
from lmvg import LMVG, hamming_distance


class GraphBackendWrapper(nn.Module):
    def __init__(self, base_model: TCAGNN, backend="nvg", n_ch=10, T=14):
        super().__init__()
        self.base = base_model
        self.backend = backend
        if backend == "lmvg":
            self.lmvg = LMVG(n_ch=n_ch, T=T)
        else:
            self.lmvg = None
        self._last_stats = {}
        self._last_hard_adj = None

    def build_graph(self, x_btc, temperature=1.0):
        if self.backend == "nvg":
            adj = _nvg_batch(x_btc)
            hard = (adj > 0.5).float().detach()
            stats = dict(
                tau_mean=0.0, tau_std=0.0,
                view_w=[1.0, 0.0, 0.0],
                sparsity=float(hard.mean().item()),
            )
            self._last_stats = stats
            self._last_hard_adj = hard
            return adj, stats
        adj, stats = self.lmvg(x_btc, temperature=temperature)
        self._last_stats = stats
        self._last_hard_adj = self.lmvg._last_adj
        return adj, stats

    def forward(self, x_btc, temperature=1.0):
        B, T, C = x_btc.shape
        adj, _ = self.build_graph(x_btc, temperature)
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

    def extract_embedding(self, x_btc, temperature=1.0):
        B, T, C = x_btc.shape
        adj, _ = self.build_graph(x_btc, temperature)
        ch_embeds = []
        for c in range(C):
            h = self.base.proj(x_btc[:, :, c : c + 1])
            h = self.base.pos(h)
            h = self.base.intra[c](h, adj[:, c])
            ch_embeds.append(h.mean(dim=1, keepdim=True))
        h_bcd = torch.cat(ch_embeds, dim=1)
        h_bcd = self.base.inter(h_bcd)
        return h_bcd.mean(dim=1)


def make_model(backend="nvg", n_ch=10, T=14):
    base = TCAGNN(n_ch=n_ch, T=T)
    return GraphBackendWrapper(base, backend=backend, n_ch=n_ch, T=T)
