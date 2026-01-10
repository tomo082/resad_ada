from typing import Tuple
import numpy as np
import torch
import torch.nn as nn

# VectorQuantizer クラスは変更なしでOKですが、念のため全文載せます
class VectorQuantizer(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly avoids costly matrix
    multiplications and allows for post-hoc remapping of indices.
    """
    def __init__(
        self,
        n_e: int,
        vq_embed_dim: int,
        beta: float,
        remap=None,
        unknown_index: str = "random",
        sane_index_shape: bool = False,
        legacy: bool = True,
    ):
        super().__init__()
        self.n_e = n_e
        self.vq_embed_dim = vq_embed_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.vq_embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.used: torch.Tensor
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(
                f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds: torch.LongTensor) -> torch.LongTensor:
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds: torch.LongTensor) -> torch.LongTensor:
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z: torch.FloatTensor, m: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, Tuple]:
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.vq_embed_dim)
        m = m.reshape(-1)
        z_normal = z_flattened[m == 0]

        min_encoding_indices = torch.argmin(torch.cdist(z_normal, self.embedding.weight), dim=1)

        z_q = self.embedding(min_encoding_indices)
        perplexity = None
        min_encodings = None

        loss = torch.mean((z_q.detach() - z_normal) ** 2) + self.beta * torch.mean((z_q - z_normal.detach()) ** 2)
        z_q: torch.FloatTensor = z_normal + (z_q - z_normal).detach()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_condebook_entry(self, z: torch.FloatTensor) -> torch.FloatTensor:
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.vq_embed_dim)
        min_encoding_indices = torch.argmin(torch.cdist(z_flattened, self.embedding.weight), dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q

# === ここを修正 ===
class MultiScaleVQ(nn.Module):
    def __init__(self, 
                 num_embeddings = 1024,
                 channels = [256, 512, 1024]): # Defaultをリスト形式の表記に
        super().__init__()
        # nn.ModuleListを使って動的に層を作成
        self.vqs = nn.ModuleList([
            VectorQuantizer(num_embeddings, c, beta=0.25, remap=None, sane_index_shape=False)
            for c in channels
        ])
    
    def forward(self, features, masks=None, train=True):
        if train:
            loss = 0
            # ループ処理に変更して全てのfeatureに対応
            for i, vq in enumerate(self.vqs):
                _, l, _ = vq(features[i], masks[i])
                loss += l
            return loss
        else:
            outputs = []
            for i, vq in enumerate(self.vqs):
                qx = vq.get_condebook_entry(features[i])
                outputs.append(qx)
            return tuple(outputs)
